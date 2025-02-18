# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from dataclasses import dataclass, field
from typing import Optional, Callable
from functools import partial
from omegaconf import II
from enum import Enum, auto
from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.dataclass import FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model

import random 

from .base import (
    MaskSeed,
    D2vModalityConfig,
    ModalitySpecificEncoder, 
    get_annealed_rate,
)

from .modules import (
    D2vDecoderConfig,
    AltBlock,
    Decoder1d,
)

from .images import (
    D2vImageConfig,
    ImageEncoder,
)

VERBOSE=False

##
from .mixit_loss import get_mixit_loss
loss_mixit = get_mixit_loss()
###
logger = logging.getLogger(__name__)

# we follow the work of data2vec 2.0 on image modality and Audio-MAE in EAT 
class Modality(Enum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()

@dataclass
class D2vModalitiesConfig(FairseqDataclass):
    image: D2vImageConfig = D2vImageConfig()
    
@dataclass
class Data2VecMultiConfig(FairseqDataclass):

    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )

    depth: int = 12
    
    # standard vision Transformer
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0
    num_heads: int = 12
    norm_eps: float = 1e-6
    norm_affine: bool = True
    encoder_dropout: float = 0.1
    post_mlp_drop: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0
    embed_dim: int = 768
    mlp_ratio: float = 4
    layer_norm_first: bool = False

    # EAT averages all Transformer block output (12 layers in total) 
    average_top_k_layers: int = field(
        default=12, metadata={"help": "how many layers to average"}
    )

    end_of_block_targets: bool = False

    # clone batch for multi-mask strategy
    clone_batch: int = 16

    # normalization for teacher Transformer layer output
    layer_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False

    # EMA settings
    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_same_dtype: bool = True
    log_norms: bool = True
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    ema_anneal_end_step: int = II("optimization.max_update")

    # In EAT, the Transformer encoder and the CNN encoder are both EMA updated
    ema_encoder_only: bool = field(
        default=True,
        metadata={
            "help": "whether to momentum update only the shared transformer encoder"
        },
    )

    max_update: int = II("optimization.max_update")

    modalities: D2vModalitiesConfig = D2vModalitiesConfig()

    shared_decoder: Optional[D2vDecoderConfig] = None

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )

    supported_modality: Optional[Modality] = None
    mae_init: bool = False

    seed: int = II("common.seed")

    skip_ema: bool = False

    # d2v_loss is the frame-level loss while cls_loss is the utterance-level loss
    cls_loss: float = 0
    recon_loss: float = 0
    d2v_loss: float = 1

    decoder_group: bool = False

    # the experiment of using dino loss instead of direct utterance loss (not included in our paper)
    utterance_level: bool = field(default=False, metadata={"help": "if true, we will add utterance-level loss to the total loss"})
    init_center_token_zero: bool = field(default=False, metadata={"help": "if true, we will initialize the centor token with zero vertors"})
    center_exp: float = field(default=0.9, metadata={"help": "this value control the exponent decay of center value's coefficient"})
    softmax_temperature_student: float = field(default=0.1, metadata={"help": "this value control the temperature of softmax function of student output in the dino loss"})
    softmax_temperature_teacher: float = field(default=0.05, metadata={"help": "this value control the temperature of softmax function in teacher output the dino loss"})


@register_model("data2vec_multi", dataclass=Data2VecMultiConfig)
class Data2VecMultiModel(BaseFairseqModel):
    def make_modality_encoder(
        self,
        cfg: D2vModalityConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases,
        task,
    ) -> ModalitySpecificEncoder:
        if cfg.type.value == Modality.IMAGE.value:
            enc_cls = ImageEncoder
        else:
            raise Exception(f"unsupported modality {cfg.type}")

        return enc_cls(
            cfg,
            embed_dim,
            make_block,
            norm_layer,
            layer_norm_first,
            alibi_biases,
            task,
        )

    def __init__(self, cfg: Data2VecMultiConfig, modalities, skip_ema=False, task=None):
        super().__init__()
        self.cfg = cfg
        self.modalities = modalities
        self.task = task

        make_layer_norm = partial(
            nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        )

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                cfg.embed_dim if dim is None else dim,
                cfg.num_heads if heads is None else heads,
                cfg.mlp_ratio,
                qkv_bias=True,
                drop=cfg.encoder_dropout,
                attn_drop=cfg.attention_dropout,
                mlp_drop=cfg.activation_dropout,
                post_mlp_drop=cfg.post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=cfg.layer_norm_first,
                ffn_targets=not cfg.end_of_block_targets,
            )

        self.alibi_biases = {}
        self.modality_encoders = nn.ModuleDict()
        
        # extract CNN encoder and CNN decoder from modified data2vec image modality (see image.py)
        for mod in self.modalities:
            mod_cfg = getattr(cfg.modalities, mod.name.lower())
            enc = self.make_modality_encoder(
                mod_cfg,
                cfg.embed_dim,
                make_block,
                make_layer_norm,
                cfg.layer_norm_first,
                self.alibi_biases,
                task,
            )
            self.modality_encoders[mod.name] = enc

        self.ema = None

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale
        self.utterance_level = cfg.utterance_level

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        dpr = np.linspace(cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth)

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])

        self.norm = None
        if cfg.layer_norm_first:
            self.norm = make_layer_norm(cfg.embed_dim)

        if self.cfg.mae_init:
            self.apply(self._init_weights)
        else:
            from fairseq.modules.transformer_sentence_encoder import init_bert_params

            self.apply(init_bert_params)

        for mod_enc in self.modality_encoders.values():
            mod_enc.reset_parameters()

        # make teacher model
        if not skip_ema:
            self.ema = self.make_ema_teacher(cfg.ema_decay)
            self.shared_decoder = (
                Decoder1d(cfg.shared_decoder, cfg.embed_dim)
                if self.cfg.shared_decoder is not None
                else None
            )
            if self.shared_decoder is not None:
                self.shared_decoder.apply(self._init_weights)

            self.recon_proj = None
            if cfg.recon_loss > 0:
                self.recon_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim//3)
                
            self.cls_proj = None
            if cfg.utterance_level:
                self.cls_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}
            if cfg.decoder_group and "decoder" in pn:
                p.param_group = "decoder"
        
        # dino loss experiment
        self.center = None
        if self.utterance_level:
            self.center_exp = cfg.center_exp
            self.soft_tem_s = cfg.softmax_temperature_student
            self.soft_tem_t = cfg.softmax_temperature_teacher
            self.center = nn.Parameter(
                    torch.zeros(1, 1, cfg.embed_dim, requires_grad=False)
                )
            if not cfg.init_center_token_zero:
                nn.init.normal_(self.center)
            elif self.center.size(1) > 1:
                nn.init.normal_(self.center[:, 1:])

        self.num_updates = 0

    def _init_weights(self, m):

        try:
            from apex.normalization import FusedLayerNorm

            fn = FusedLayerNorm
        except:
            fn = nn.LayerNorm

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, fn):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def make_ema_teacher(self, ema_decay):
        ema_config = EMAModuleConfig(
            ema_decay=ema_decay,
            ema_fp32=True,
            log_norms=self.cfg.log_norms,
            add_missing_params=False,
        )

        model_copy = self.make_target_model()

        return EMAModule(
            model_copy,
            ema_config,
            copy_model=False,
        )

    # teacher model (with independent CNN encoder and Transformer encoder)
    def make_target_model(self):
        logger.info("making target model")

        model_copy = Data2VecMultiModel(
            self.cfg, self.modalities, skip_ema=True, task=self.task
        )

        if self.cfg.ema_encoder_only:
            model_copy = model_copy.blocks
            for p_s, p_t in zip(self.blocks.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)
        else:
            for p_s, p_t in zip(self.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)

            for mod_enc in model_copy.modality_encoders.values():
                mod_enc.decoder = None
                if not mod_enc.modality_cfg.ema_local_encoder:
                    mod_enc.local_encoder = None
                    mod_enc.project_features = None

        model_copy.requires_grad_(False)
        return model_copy

    # teacher model updated with EMA
    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is not None and (
            (self.num_updates == 0 and num_updates > 1)
            or self.num_updates >= num_updates
        ):
            pass
        elif self.training and self.ema is not None:
            ema_weight_decay = None
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay, weight_decay=ema_weight_decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.blocks if self.cfg.ema_encoder_only else self)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        k = prefix + "_ema"
        if self.ema is not None:
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        elif k in state_dict:
            del state_dict[k]

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: Data2VecMultiConfig, task=None):
        """Build a new model instance."""
        if task is None or not hasattr(task, "supported_modalities"):
            modalities = (
                [cfg.supported_modality]
                if cfg.supported_modality is not None
                else [
                    Modality.AUDIO,
                    Modality.IMAGE,
                    Modality.TEXT,
                ]
            )
        else:
            modalities = task.supported_modalities
        return cls(cfg, modalities, task=task, skip_ema=cfg.skip_ema)

    def prepare_dropped_inputs_for_teacher_after_pos(self, patch_inp, mask_indices,no_mix_region_size,patch_size=16,num_extra_tokens=1):
        """
        Args:
            patch_inp (Tensor): INPUT after patch embedding. (batch_size,513,768)
            mask_indices (List): The indices where mixing is not done. all are multiples of 16

        Returns:
            Tensor: rejoined (Tensor): The input tensor with the mixed region. (A1, A2, ...)

        AUDIO 1: =============idx1 XXXXXXXXXXXXXX idx1+no_mix_region_size=============idx2 XXXXXXXXXXXXXX idx2+no_mix_region_size=============
                  part1                                                        part2                                                part3
        RETURN : CONCAT(part1, part2, part3)
        """
        batch_size = patch_inp.shape[0]


        ## BRINGING TO (batch_size,768,T,F) 
        assert patch_inp.shape[1:] == (513,768), f"unexpected patch_inp shape: {patch_inp.shape}"
        cls_token_to_restore_later = patch_inp[:,0,:].unsqueeze(1) ## (batch_size,1,768)
        patch_inp_without_cls = patch_inp[:,num_extra_tokens:,:] ## remove cls token num_cls_tokens = 1
        patch_inp_without_cls = patch_inp_without_cls.permute(0,2,1) ## (batch_size,768,512)
        patch_inp_without_cls = patch_inp_without_cls.reshape(batch_size,768,64,8) ## (batch_size,768,32,16)

        ### CONVERT MIX_INDICES TO PATCH INDICES by dividing by patch_size(16)
        idx1 = mask_indices[0]
        idx2 = mask_indices[1]

        idx1 = idx1//patch_size
        idx2 = idx2//patch_size
        no_mix_region_size = no_mix_region_size//patch_size

        ### TAKE FORIGN AUDIO A1, AND CONCAT ONLY THE PARTS THAT ARE MIXED WITH A2
        part1 = patch_inp_without_cls[:,:,:idx1,:]
        part2 = patch_inp_without_cls[:,:,idx1+no_mix_region_size:idx2,:]
        part3 = patch_inp_without_cls[:,:,idx2+no_mix_region_size:,:]

        rejoined = torch.cat((part1, part2, part3), dim=-2)
        # print(f"rejoined shape: {rejoined.shape}")
        assert rejoined.shape[1:] == (768,34,8), f"unexpected rejoined shape: {rejoined.shape}"

        ### BRING BACK TO (batch_size,272,768)
        rejoined = rejoined.reshape(batch_size,768,-1) ## (batch_size,768,272)
        rejoined = rejoined.permute(0,2,1) ## (batch_size,272,768)

        assert rejoined.shape[1:] == (272,768), f"unexpected rejoined shape: {rejoined.shape}"

        ### CONCAT THE CLASS TOKEN BACK
        rejoined = torch.cat((cls_token_to_restore_later,rejoined),dim=1) ## (batch_size,273,768)
        return rejoined

    def forward(
        self,
        source,
        target=None,
        id=None,
        mode=None,
        padding_mask=None,
        mask=True,
        features_only=False,
        force_remove_masked=False,
        remove_extra_tokens=True,
        precomputed_mask=None,
    ):
        if mode is None:
            assert self.cfg.supported_modality is not None
            mode = self.cfg.supported_modality

        if isinstance(mode, Modality):
            mode = mode.name


        feature_extractor = self.modality_encoders[mode]

        mask_seeds = None
        if id is not None:
            exit("why here? id is not None")
            mask_seeds = MaskSeed(seed=self.cfg.seed, update=self.num_updates, ids=id)
        if VERBOSE:
            print("mode: ", mode)
            print("source shape: ", source.shape)
            # print('feature_extractor: ', feature_extractor)
        
        ### MIXING SPECTS
        ### SAY, MIXING A1(FOREIGN) into A2 PARTIALLY
        orig_source = source.clone() ## (6,1,1024,128)
        if source.shape != (12,1,1024,128):
            print(f"unexpected source shape: {source.shape}")

        ### BRING A1(FOREIGN) TO SAME INDEX AS A2 (i.e. index 1)
        roll1 = torch.roll(source, shifts=1, dims=0)


        ###GET INDICES WHERE NOT TO MIX. MAKE SURE THEY ARE MULTIPLES OF 16
        no_mix_region_size = 240
        idx_1 = random.choice([i for i in range(0, 501, 16)]) ## random multiple of 16 between 0 and 500
        idx_2 = random.choice([i for i in range(idx_1+no_mix_region_size, 1023-no_mix_region_size, 16)]) ## random multiple of 16 between 0 and 500
        mix_indices = [idx_1, idx_2]

        # Set the region from each index to index+250 to 0.0
        for idx in mix_indices:
            roll1[:, :, idx:idx+no_mix_region_size, :] = float('-inf') ## no mixing in that region, original spect will be used
        """
        MIXING A1(FOREIGN) into A2 PARTIALLY
        AUDIO 1 in roll1: =============idx1 -inf-inf-inf idx1+no_mix_region_size=============idx2 -inf-inf-inf idx2+no_mix_region_size=============
                              part1                                                  part2                                                part3
        AUDIO 2 in source:==========================================================================================================================
        """

        mix_source = torch.max(source, roll1)

        """
        MIXED SPECT mix_source : A2A1-A2A1-A2A1--idx_1----A2-A2-A2-A2----idx_1+240--A2A1-A2A1-A2A1--idx_2----A2-A2-A2-A2----idx_2+240--A2A1-A2A1-A2A1
        
        """

        # assert mix_source.shape == (12,1,1024,128) ## last batch will throw error
        if mix_source.shape != (12,1,1024,128): ## residual batch will throw warning, but can be ignored
            print(f"unexpected mix_source shape: {mix_source.shape}")

        ### MAKEING MAE MASKS
        ##  CONCAT UNMIXED AND MIXED SPECTS TO MAKE A BATCH OF x2 = 24
        source = torch.cat((source, mix_source), dim=0) ## residual batch will throw warning, but can be ignored
        if source.shape != (24,1,1024,128): ## residual batch will throw warning, but can be ignored
            print(f"unexpected source shape: {source.shape}")

        ### FOR A2A1 USE THE SAME MASK AS A2; No specific reason, just to keep it simple
        precomputed_mask = torch.cat((precomputed_mask, precomputed_mask), dim=0) ## orig_source first then mix_source
        # assert precomputed_mask.shape == (192, 512) ## last batch will throw error
        if precomputed_mask.shape != (192, 512): #residual batch will throw warning, but can be ignored
            print(f"unexpected precomputed_mask shape: {precomputed_mask.shape}")

        #### END OF MIXING SPECTS
        
        if VERBOSE:
            print("after mixing source shape: ", source.shape)
            print("precomputed_mask shape: ", precomputed_mask.shape)

        #### end mixing spects
        # extract (unmasked) features using CNN encoder
        assert padding_mask is None
        assert mask is True
        assert mask_seeds is None
        removed_masked_ = not features_only or force_remove_masked
        assert removed_masked_ == True

        #### PATCH EMBEDDING and MASKING FOR STUDENT
        extractor_out = feature_extractor(
            source,
            padding_mask,
            mask,
            remove_masked=not features_only or force_remove_masked,
            clone_batch=self.cfg.clone_batch if not features_only else 1,
            mask_seeds=mask_seeds,
            precomputed_mask=precomputed_mask,
        )

        # x in shape ( batch_size * clone batch, patch_frame(64) * patch_freqency(8) * unmask_ratio(0.2) + 1(cls_token), 768(feature dimension) )
        # EAT does not employ the ablibi mechanism in Transformer
        x = extractor_out["x"]
        # print("extracter output shape: ", x.shape)
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        #### sanity check
        assert masked_padding_mask is None
        assert masked_alibi_bias is None
        assert alibi_scale is None
        ### end sanity check

        if VERBOSE:
            print("extracter output shape: ", x.shape)

        if self.dropout_input is not None:
            x = self.dropout_input(x)
        
        if VERBOSE:
            print("self.blocks ",self.blocks)

        # standard Transformer (for student encoder)
        layer_results = []
        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.cfg.layerdrop == 0
                or (np.random.random() > self.cfg.layerdrop)
            ):
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                # if VERBOSE:
                #     print(f"block {i} input ", x.shape)
                assert masked_padding_mask is None
                assert ab is None

                x, lr = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                )
                # if VERBOSE:
                #     print(f"block {i} output ", x.shape)

                if features_only:
                    layer_results.append(lr)

        if self.norm is not None:
            x = self.norm(x)

        # extract features for fine-tuning
        if features_only:
            if remove_extra_tokens:
                x = x[:, feature_extractor.modality_cfg.num_extra_tokens :]
                if masked_padding_mask is not None:
                    masked_padding_mask = masked_padding_mask[
                        :, feature_extractor.modality_cfg.num_extra_tokens :
                    ]

            return {
                "x": x,
                "padding_mask": masked_padding_mask,
                "layer_results": layer_results,
                "mask": encoder_mask,
            }

        # decode features merged with masked tokens, dx in shape (batch_size * clone_batch, patch, 768)
        xs = []
        if self.shared_decoder is not None:
            exit("why here? self.shared_decoder is not None ")
            dx = self.forward_decoder(
                x,
                feature_extractor,
                self.shared_decoder,
                encoder_mask,
            )
            xs.append(dx)
        if feature_extractor.decoder is not None:  ## @change no decoder now
            dx = self.forward_decoder(
                x,
                feature_extractor,
                feature_extractor.decoder,
                encoder_mask,
            )
            xs.append(dx)
            orig_x = x
            ### pass

            

        assert len(xs) > 0 ## @change no decoder now
        ### @change no decoder now

        p = next(self.ema.model.parameters())
        device = x.device
        dtype = x.dtype
        ema_device = p.device
        ema_dtype = p.dtype

        if not self.cfg.ema_same_dtype:
            dtype = ema_dtype

        if ema_device != device or ema_dtype != dtype:
            logger.info(f"adjusting ema dtype to {dtype} and device to {device}")
            self.ema.model = self.ema.model.to(dtype=dtype, device=device)
            ema_dtype = dtype

            def to_device(d):
                for k, p in d.items():
                    if isinstance(d[k], dict):
                        to_device(d[k])
                    else:
                        d[k] = p.to(device=device)

            to_device(self.ema.fp32_params)
        tm = self.ema.model

        # encode audio spectrogram using teacher model
        with torch.no_grad():
            tm.eval()

            if self.cfg.ema_encoder_only:
                exit("why here? ema_encoder_only")
                assert target is None
                ema_input = extractor_out["local_features"]
                ema_input = feature_extractor.contextualized_features(
                    ema_input.to(dtype=ema_dtype),
                    padding_mask,
                    mask=False,
                    remove_masked=False,
                )
                ema_blocks = tm
            else:
                ema_blocks = tm.blocks
                if feature_extractor.modality_cfg.ema_local_encoder:
                    inp = (
                        target.to(dtype=ema_dtype)
                        if target is not None
                        else source.to(dtype=ema_dtype)
                    )
                    assert target is None,'target not Noone'
                    
                    #### CONCAT OF NON MIXED(BS/2) and MIXED AUDIO(BS/2) - (24,1,1024,128)
                    inp = source 
                    assert padding_mask is None

                    if VERBOSE:
                        print("inp shape: ", inp.shape)
                    ema_input = tm.modality_encoders[mode](
                        inp,
                        padding_mask,
                        mask=False,
                        remove_masked=False,
                    )

                    #### UNMIXED AUDIO ONLY - HALF OF BATCH(24/2) - (12,1,1024,128)
                    inp2 = orig_source 

                    if VERBOSE:
                        print("inp shape: ", inp.shape)
                    
                    ema_input2 = tm.modality_encoders[mode](
                        inp2,
                        padding_mask,
                        mask=False,
                        remove_masked=False,
                    )


                else:
                    exit('why here? not feature_extractor.modality_cfg.ema_local_encoder')
                    assert target is None
                    ema_input = extractor_out["local_features"]
                    ema_feature_enc = tm.modality_encoders[mode]
                    ema_input = ema_feature_enc.contextualized_features(
                        ema_input.to(dtype=ema_dtype),
                        padding_mask,
                        mask=False,
                        remove_masked=False,
                    )

            ema_padding_mask = ema_input["padding_mask"]
            ema_alibi_bias = ema_input.get("alibi_bias", None)
            ema_alibi_scale = ema_input.get("alibi_scale", None)
            ema_input = ema_input["x"]

            ### verify whats None
            assert ema_padding_mask is None, 'why here? ema_padding_mask is not None'
            assert ema_alibi_bias is None, 'why here? ema_alibi_bias is not None'
            assert ema_alibi_scale is None, 'why here? ema_alibi_scale is not None'

            ## extra batch dropped audios
            ema_padding_mask2 = ema_input2["padding_mask"]
            ema_alibi_bias2 = ema_input2.get("alibi_bias", None)
            ema_alibi_scale2 = ema_input2.get("alibi_scale", None)
            ema_input2 = ema_input2["x"]

            assert ema_padding_mask2 is None, 'why here? ema_padding_mask2 is not None'
            assert ema_alibi_bias2 is None, 'why here? ema_alibi_bias2 is not None'
            assert ema_alibi_scale2 is None, 'why here? ema_alibi_scale2 is not None'

            
            #### DROP UNMIXED REGION FROM A1 in the MIXED-AUDIO A2A1
            ema_input2 = self.prepare_dropped_inputs_for_teacher_after_pos(patch_inp=ema_input2,mask_indices=mix_indices,no_mix_region_size=no_mix_region_size)


            # TEACHER MODEL FORWARD PASS of CONCAT OF NON MIXED(BS/2) and MIXED AUDIO(BS/2) - (24,1,1024,128)
            # ema_input in shape (batch_size, patch + 1(cls_token), feature_dimension)
            y = []
            ema_x = []
            extra_tokens = feature_extractor.modality_cfg.num_extra_tokens
            for i, blk in enumerate(ema_blocks):  
                ab = ema_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        ema_alibi_scale[i]
                        if ema_alibi_scale.size(0) > 1
                        else ema_alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                ema_input, lr = blk(
                    ema_input,
                    padding_mask=ema_padding_mask,
                    alibi_bias=ab,
                )
                y.append(lr[:, extra_tokens:])
                ema_x.append(ema_input[:, extra_tokens:])


            ### TEACHER MODEL FORWARD PASS of DROPPED UNMIXED AUDIO - (12,1,1024,128)
            y2 = []
            ema_x2 = []
            extra_tokens = feature_extractor.modality_cfg.num_extra_tokens
            for i, blk in enumerate(ema_blocks):  
                ab2 = ema_alibi_bias2
                if ab2 is not None and alibi_scale is not None:
                    scale = (
                        ema_alibi_scale2[i]
                        if ema_alibi_scale2.size(0) > 1
                        else ema_alibi_scale2.squeeze(0)
                    )
                    ab2 = ab2 * scale.type_as(ab2)

                assert ab2 is None , 'why here? ab2 is not None'
                assert ema_padding_mask2 is None, 'why here? ema_padding_mask2 is not None'

                ema_input2, lr2 = blk(
                    ema_input2,
                    padding_mask=ema_padding_mask2,
                    alibi_bias=ab2,
                )

                # print(f"lr2 shape: {lr2.shape}, ema_input2: {ema_input2.shape}")

                y2.append(lr2[:, extra_tokens:])
                ema_x2.append(ema_input2[:, extra_tokens:])
        # EAT utilize total 12 Transformer block layer output average as target 
        # y: CONCAT OF NON MIXED(BS/2) and MIXED AUDIO(BS/2) - ALL 12 LAYERS 
        # y_non_mixed: NON MIXED AUDIO ONLY - HALF OF BATCH(24/2)- ALL 12 LAYERS

        #### GET NON MIXED AUDIO ONLY - HALF OF BATCH(24/2)- ALL 12 LAYERS
        y_non_mixed = [t_i.clone() for t_i in y]
        y_non_mixed = [t_i.chunk(2,dim=0)[0] for t_i in y_non_mixed]

        """
        Targets required
        here orig refers to EAT (as in original method) not the original audio
        1) for EAT patch-level loss:
            y_orig_patch: Concat of original+partial mixed audio - one batch - (24,512,768)

        2) EAT cls loss:
            y_orig_cls: created using original audio + partial mixed audio - one batch - (24,1,768)

        3) MixLoss cls loss: XXXXXXXX DROPPED
            y_mix_cls: Created using non-mixed audio + dropped audio - one batch - (12,1,768)
            
        4) MixLoss patch-level loss:
            mixed_target_patch: Created using non-mixed audio + dropped audio - one batch - (12,512,768)

        """
        ### EAT PATCH/LOCAL LOSS

        ## here orig refers to EAT (as in original method) not the original audio
        assert self.average_top_k_layers == 12, 'why here? self.average_top_k_layers is not 12'
        y_for_orig_patch = [t_i.clone() for t_i in y]  ### safty from inplace operation

        y_orig_patch = self.make_targets(y_for_orig_patch, self.average_top_k_layers) ## for EAT patch-level loss

        ### EAT CLS/GLOBAL LOSS
        y_for_cls = y ## not using y after this point safe to 
        y_orig_cls = self.make_targets(y_for_cls, 1) ## for EAT cls loss

        ###XXXXXXXXXXXXXXX MIXLOSS CLS/GLOBAL LOSS XXX NOT USING FOR NOW
        y_non_mix_for_cls = [t_i.clone() for t_i in y_non_mixed]  ### safty from inplace operation
        y2_for_cls = [t_i.clone() for t_i in y2]  ### safty from inplace operation

        ### NOT USING MIX CLS LOSS FOR NOW
        y_mix_cls = self.get_cls_target_for_mix(y_non_mixed=y_non_mix_for_cls,y_dropped=y2_for_cls,num_layers=1) ## for MixLoss cls loss
        ###XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

        ### MIXLOSS PATCH/LOCAL LOSS
        y_non_mix_for_patch = y_non_mixed ## not using y_non_mixed after this point safe to
        y2_for_patch = y2 ## not using y2 after this point safe to
        y_mix_patch = self.get_patch_target_for_mix(y_non_mixed=y_non_mix_for_patch,y_dropped=y2_for_patch,num_layers=self.average_top_k_layers,mix_indices=mix_indices,no_mix_region_size=no_mix_region_size)## for MixLoss patch-level loss



        if y_orig_patch.shape != (24,512,768):
            print(f"unexpected y_orig_patch shape: {y_orig_patch.shape}")
        if y_orig_cls.shape != (24,512,768):
            print(f"unexpected y_orig_cls shape: {y_orig_cls.shape}")
        if y_mix_patch.shape != (12,512,768):
            print(f"unexpected y_mix_patch shape: {y_mix_patch.shape}")

        if y_mix_cls.shape != (12,1,768):
            print(f"unexpected y_mix_cls shape: {y_mix_cls.shape}")
        orig_targets_y_orig_cls = y_orig_cls





        # multiply the target value according to the number of clone batch
        if self.cfg.clone_batch > 1:
            # y = y.repeat_interleave(self.cfg.clone_batch, 0)
            y_orig_patch = y_orig_patch.repeat_interleave(self.cfg.clone_batch, 0)
            y_mix_patch = y_mix_patch.repeat_interleave(self.cfg.clone_batch, 0)


        # extract values in masked position to make prediction
        masked = encoder_mask.mask.unsqueeze(-1)
        masked_b = encoder_mask.mask.bool()

        y_orig_patch = y_orig_patch[masked_b]  ## for EAT PATCH/LOCAL LOSS

        masked_b_non_mixed, masked_b_mixed = masked_b.chunk(2, dim=0)

        if not torch.equal(masked_b_non_mixed, masked_b_mixed): ## they should be same
            print(f"unexpected  should be same because i concat precomuted_mask twice")

        y_mix_patch = y_mix_patch[masked_b_mixed] ## for MIXLOSS PATCH/LOCAL LOSS

        # y_roll1 = y_roll1[masked_b] ## for decoder loss
        """
        here dont roll masked_b, because we want to keep the mask 
        used for mixed audio 1 and roll1 audio(audio2) in student
        """
        if VERBOSE:
            print("masked_b shape: ", masked_b.shape)
            # print("y shape: ", y.shape)

        ###STUDENT OUTPUTS
        assert len(xs) == 1, 'expect only one decoder output'
        stu_out = xs[0] ## stu_out shape:  torch.Size([192, 512, 768])
        # assert stu_out.shape == (192,512,768), f"stu_out shape: {stu_out.shape}"

        ### GET STUDENT OUTPUTS FOR UNMIXED AND MIXED HALFS OF THE BATCH
        stu_out_non_mixed_not_using, stu_out_mixed = stu_out.chunk(2, dim=0)
        
        if stu_out_mixed.shape != (96,512,768): ### residual batch will throw warning, but can be ignored
            print(f"unexpected stu_out_mixed shape: {stu_out_mixed.shape}")

        xs_mixed = [stu_out_mixed] ## for MIXLOSS PATCH/LOCAL LOSS; Keep in list for consistency with xs


        ### GET STUDENT OUTPUTS FOR MAE MASKED POSITIONS
        if xs[0].size(1) == masked_b.size(1):
            xs = [x[masked_b] for x in xs]
            xs_mixed = [x[masked_b_mixed] for x in xs_mixed]
            assert len(xs) == len(xs_mixed) == 1, 'expecting one decoder output'
        else:
            exit('why here? xs[0].size(1) != masked_b.size(1) @change')
            xs = [x.reshape(-1, x.size(-1)) for x in xs]

        ## end no decoder now

            

        sample_size = masked.sum().long() #78720
        # print("sample_size: ", sample_size)

        # sample_size_mixed = sample_size // 2

        result = {
            "losses": {},
            "sample_size": sample_size,
        }

        sample_size = result["sample_size"]

        # EAT employ utterance-level loss by using mean pooling in patch dimension
        if self.cfg.cls_loss > 0 and not self.utterance_level: #### loss 1: utterance-level loss
            
            if VERBOSE:
                print('d2v_loss self.utterance_level')
                # print("orig_targets ",orig_targets.shape)
            

            assert extra_tokens > 0
            cls_target_y_orig_cls = orig_targets_y_orig_cls.mean(dim=1)
            cls_target_y_feat_mix_cls = y_mix_cls ## already mean pooled in fn get_cls_target_for_mix

            # cls_target_roll1 = torch.roll(cls_target, shifts=1, dims=0) ### @change roll approach

            if self.cfg.clone_batch > 1:
                cls_target_y_orig_cls = cls_target_y_orig_cls.repeat_interleave(self.cfg.clone_batch, 0)
                cls_target_y_feat_mix_cls = cls_target_y_feat_mix_cls.repeat_interleave(self.cfg.clone_batch, 0)
                ##@change
                # cls_target_roll1 = cls_target_roll1.repeat_interleave(self.cfg.clone_batch, 0)

            # assert torch.equal(x,to_test), 'x and to_test should be same'
            ### @change
            # cls_pred = x[:, extra_tokens - 1]
            cls_pred = x[:, :extra_tokens]

            # print("cls_pred shape: ", cls_pred.shape)
            # cls_pred_non_mixed_not_using, cls_pred_mixed = cls_pred.chunk(2, dim=0)

            # cls_target_avg = (cls_target + cls_target_roll1) / 2.0

            eat_cls_loss = self.d2v_loss(cls_pred, cls_target_y_orig_cls) * (
                self.cfg.cls_loss * sample_size)

            # mix_cls_loss = self.d2v_loss(cls_pred_mixed, cls_target_y_feat_mix_cls) * (
            #     self.cfg.cls_loss * sample_size) ### keeping the same sample size for cls loss
            # mix_cls_loss = mix_cls_loss * 2.0 ## scale to match the sample size of EAT cls loss

            # print(f"eat_cls_loss: {eat_cls_loss.shape}, mix_cls_loss: {mix_cls_loss.shape}") ## eat_cls_loss: torch.Size([192, 768]), mix_cls_loss: torch.Size([96, 768])  
            result["losses"]["eat_cls_loss"] = eat_cls_loss
            # result["losses"]["mix_cls_loss"] = mix_cls_loss
            
    
            # result["losses"]["cls"] = self.d2v_loss(cls_pred_mixed, cls_target_y_feat_mix_cls) * (
            #     self.cfg.cls_loss * sample_size
            # )

        # dino loss experiment
        if self.cfg.cls_loss > 0 and self.utterance_level:
            exit('why here? dino loss self.utterance_level')
            assert extra_tokens > 0
            cls_target = orig_targets.mean(dim=1)
            if self.cfg.clone_batch > 1:
                cls_target = cls_target.repeat_interleave(self.cfg.clone_batch, 0)  #(btz*clone,1,768)
            cls_pred = x[:, extra_tokens - 1]
            cls_target = cls_target - self.center
            
            cls_pred = cls_pred.squeeze(dim=1)
            cls_target = cls_target.squeeze(dim=1)
            
            result["losses"]["cls"] = self.dino_loss(cls_pred, cls_target) * (
                self.cfg.cls_loss * sample_size
            )
            
            self.center = self.center_exp * self.center + (1 - self.center_exp) * (cls_target.mean(dim=0))
        
        if self.cfg.recon_loss > 0:
            exit("why here? recon loss")
            
            with torch.no_grad():
                target = feature_extractor.patchify(source)  #(btz,1,512,16*16)
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5   #(btz,1,512,1)

                if self.cfg.clone_batch > 1:
                    target = target.repeat_interleave(self.cfg.clone_batch, 0)  #(btz*clone_btz,1,512,1)

                if masked_b is not None:
                    target = target[masked_b]

            recon = xs[0]
            if self.recon_proj is not None:
                recon = self.recon_proj(recon)

            result["losses"]["recon"] = (
                self.d2v_loss(recon, target.float()) * self.cfg.recon_loss
            )



    


        ### @change no decoder now
        if self.cfg.d2v_loss > 0: #### loss 2: frame-level loss
            assert len(xs) == 1, 'why here? one decoder output expected @change'
            assert len(xs_mixed) == 1, 'why here? one decoder output expected @change'
            # for i, x in enumerate(xs):
            #     reg_loss = self.d2v_loss(x, y)
            #     n = f"{mode}_regression_{i}" if len(xs) > 1 else f"{mode}_regression"
            #     result["losses"][n] = reg_loss * self.cfg.d2v_loss

            assert xs[0].shape == y_orig_patch.shape, f"xs[0] shape: {xs[0].shape}, y_orig_patch shape: {y_orig_patch.shape}"
            assert xs_mixed[0].shape == y_mix_patch.shape, f"xs_mixed[0] shape: {xs_mixed[0].shape}, y_mix_patch shape: {y_mix_patch.shape}"

            eat_patch_loss = self.d2v_loss(xs[0], y_orig_patch)
            mix_patch_loss = self.d2v_loss(xs_mixed[0], y_mix_patch)
            mix_patch_loss = mix_patch_loss * 2.0 ## scale to match the sample size of EAT patch-level loss

            result["losses"]["eat_patch_loss"] = eat_patch_loss
            result["losses"]["mix_patch_loss"] = mix_patch_loss

            y = y_orig_patch ## for remaning stuff

        
        # compute state for logs
        suffix = "" if len(self.modalities) == 1 else f"_{mode}"
        with torch.no_grad():
            if encoder_mask is not None:
                result["masked_pct"] = 1 - (
                    encoder_mask.ids_keep.size(1) / encoder_mask.ids_restore.size(1)
                )
            for i, x in enumerate(xs):
                n = f"pred_var{suffix}_{i}" if len(xs) > 1 else f"pred_var{suffix}"
                result[n] = self.compute_var(x.float())
            if self.ema is not None:
                for k, v in self.ema.logs.items():
                    result[k] = v

            y = y.float()
            result[f"target_var{suffix}"] = self.compute_var(y)

            if self.num_updates > 5000:
                if result[f"target_var{suffix}"] < self.cfg.min_target_var:
                    logger.error(
                        f"target var is {result[f'target_var{suffix}'].item()} < {self.cfg.min_target_var}, exiting ({mode})"
                    )
                    raise Exception(
                        f"target var is {result[f'target_var{suffix}'].item()} < {self.cfg.min_target_var}, exiting ({mode})"
                    )

                for k in result.keys():
                    if k.startswith("pred_var") and result[k] < self.cfg.min_pred_var:
                        logger.error(
                            f"{k} is {result[k].item()} < {self.cfg.min_pred_var}, exiting ({mode})"
                        )
                        raise Exception(
                            f"{k} is {result[k].item()} < {self.cfg.min_pred_var}, exiting ({mode})"
                        )

            result["ema_decay"] = self.ema.get_decay() * 1000

        return result

    def forward_decoder(
        self,
        x,
        feature_extractor,
        decoder,
        mask_info,  
    ):
        if VERBOSE:
            print("feature_extractor.decoder_input ",x.shape)
        x = feature_extractor.decoder_input(x, mask_info)
        x = decoder(*x)

        if VERBOSE:
            print('feature_extractor.decoder_output ',x.shape)

        return x

    def d2v_loss(self, x, y):
        # x = x.view(-1, x.size(-1)).float()
        # y = y.view(-1, x.size(-1))
        x = x.reshape(-1, x.size(-1)).float()
        y = y.reshape(-1, x.size(-1))
        assert x.dtype == y.dtype, 'x and y should have same dtype'

        if VERBOSE:
            print("in d2v_loss: x shape: ", x.shape, " y shape: ", y.shape)

        if self.loss_beta == 0:
            loss = F.mse_loss(x, y, reduction="none")
        else:
            loss = F.smooth_l1_loss(x, y, reduction="none", beta=self.loss_beta)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(x.size(-1))

        reg_loss = loss * scale

        return reg_loss

    def get_cls_target_for_mix(self,y_non_mixed,y_dropped,num_layers):
        if num_layers != 1:
            print(f"unexpected num_layers in get_cls_target_for_mix : {num_layers}")
        y_non_mixed = self.make_targets(y_non_mixed, num_layers) ## [A1,A2,A3,...,A12] (12,512,768)

        y_dropped = self.make_targets(y_dropped, num_layers) ### [A1,A2,A3,...,A12]
        y_dropped = torch.roll(y_dropped, shifts=1, dims=0) ## [A12,A1,A2,...,A11] (12,512,768)

        y_non_mixed = y_non_mixed.mean(dim=1) ## (12,768)
        y_dropped = y_dropped.mean(dim=1) ## (12,768)
        mixed_target = (y_non_mixed + y_dropped) / 2.0 ## (12,768)
        mixed_target = mixed_target.unsqueeze(1) ## (12,1,768)

        assert mixed_target.shape[1:] == (1,768), f"mixed_target shape: {mixed_target.shape}"

        return mixed_target ###  mixed_target for Mix Cls Loss

    def get_patch_target_for_mix(self,y_non_mixed,y_dropped,num_layers,mix_indices,no_mix_region_size,patch_size=16):
        """
        AUDIO 2(y_non_mixed): ===========================================================================================================

        AUDIO 1(y_dropped): CONCAT(====== ======= ========)
                                    part1  part2    part3

        Args:
            y_non_mixed: 12 layer output for non mixed half of batch #[(12,512,768)......]
            y_dropped:   12 layer output from SECOND Forward with the FOREIGN audio(That is A1 For A2) #[(12,273,768)......]
            mix_indices: indices of A2 where we didnt mix audio 1
            no_mix_region_size: indices of A2 where we didnt mix audio 1

        NOTE: BATCH_SIZE: 12

        """
        if num_layers != 12:
            print(f"unexpected num_layers in get_patch_target_for_mix : {num_layers}")
        
        ### LAYER AVERAGING
        y_non_mixed = self.make_targets(y_non_mixed, num_layers) ## [A1,A2,A3,...,A12] OUT:(BS:12,512,768)

        y_dropped = self.make_targets(y_dropped, num_layers) ### [A1,A2,A3,...,A12]
        
        ### ROLLING to MAKE FORIGN AUDIO(A1) to SAME POSITION AS A2 (that is index 1)
        y_dropped = torch.roll(y_dropped, shifts=1, dims=0) ## [A12,A1,A2,...,A11] (12,512,768)

        #### BRING BOTH TO (BS:12,768,T,F) FORMAT FOR A2: (12,768,512) and A1(DROPPED/FOREIGN): (12,768,272)
        batch_size = y_non_mixed.shape[0]
        y_non_mixed = y_non_mixed.permute(0,2,1) ## (12,768,512)
        assert y_non_mixed.shape[1:] == (768,512), f"y_non_mixed shape: {y_non_mixed.shape}"
        y_non_mixed = y_non_mixed.view(batch_size,768,64,8) ## (12,768,64,8)

        y_reconstructed = y_non_mixed.clone()

        y_dropped = y_dropped.permute(0,2,1) ## (12,768,272)
        y_dropped = y_dropped.view(batch_size,768,-1,8) ## (12,768,34,8)

        ### CONVERT MIX_INDICES TO PATCH INDICES by dividing by patch_size(16)
        idx_1 = mix_indices[0]//patch_size
        idx_2 = mix_indices[1]//patch_size
        no_mix_region_size = no_mix_region_size//patch_size

        idx_2_on_dropped = idx_2 - no_mix_region_size


        ### CUT into parts, part1, part2, part3. NOTHING TO DROP HERE

        part1 = y_dropped[:,:,:idx_1,:]
        part2 = y_dropped[:,:,idx_1:idx_2_on_dropped,:]
        part3 = y_dropped[:,:,idx_2_on_dropped:,:]

        ### TAKE A COPY OF A2(NON-MIXED) and INSERT PART1, PART2, PART3 REGIONS OF MIXING, OTHER REGIONS ARE SAME AS A2(NON-MIXED)
        y_reconstructed[:,:,:idx_1,:] = part1
        y_reconstructed[:,:,idx_1+no_mix_region_size:idx_2,:] = part2
        y_reconstructed[:,:,idx_2+no_mix_region_size:,:] = part3


        #### A2(NON-MIXED) to (12,512,768)
        y_non_mixed = y_non_mixed.view(batch_size,768,512)
        y_non_mixed = y_non_mixed.permute(0,2,1) ## (12,512,768)

        ### A1(FORIGEN AUDIO) to (12,512,768)
        y_reconstructed = y_reconstructed.view(batch_size,768,512)
        y_reconstructed = y_reconstructed.permute(0,2,1) ## (12,512,768)

        ### AVERAGE A1 AND A2
        mixed_target = (y_non_mixed + y_reconstructed) / 2.0 ## (12,512,768)

        assert mixed_target.shape[1:] == (512,768), f"mixed_target shape: {mixed_target.shape}"

        return mixed_target 
    
    def dino_loss(self,s,t):
        t = t.detach()
        s = F.softmax(s/self.soft_tem_s,dim=1)
        t = F.softmax((t-self.center)/self.soft_tem_t,dim=1)
        return - (t * torch.log(s)).sum(dim=1).mean()


    def feature_space_mix(self,y):
        y_roll1 = torch.roll(y, shifts=1, dims=0)
        y = (y + y_roll1) / 2.0
        return y
    # will roll1  average top-k layers output from teacher model
    def make_targets_split_roll1_avg(self, y, num_layers):

        with torch.no_grad():
            target_layer_results = y[-num_layers:]

            if type(target_layer_results) is not list: ## when num_layers = 1
                target_layer_results = [target_layer_results]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BTC -> BCT
                ]
                permuted = True
            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]
            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]
            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]
            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]

        y = target_layer_results[0].float()
        for tl in target_layer_results[1:]:
            y.add_(tl.float())
        y = y.div_(len(target_layer_results))

        y_orig = y.clone()

        y_non_mix,y_mix = y.chunk(2, dim=0) ### select y_non_mix drop second half(y_mix)

        y_feat_mix = self.feature_space_mix(y_non_mix)


        if self.cfg.layer_norm_targets:
            # print("self.cfg.layer_norm_targets")
            # y = F.layer_norm(y, y.shape[-1:])
            y_orig = F.layer_norm(y_orig, y.shape[-1:])
            y_feat_mix = F.layer_norm(y_feat_mix, y_feat_mix.shape[-1:])


        if self.cfg.instance_norm_targets:
            exit("self.cfg.instance_norm_targets")
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)


        # assert y_orig.shape == (24,512,768), f"y_orig shape {y_orig.shape}"
        # assert y_feat_mix.shape == (12,512,768), f"y_feat_mix shape {y_feat_mix.shape}"
        if y_orig.shape != (24,512,768):
            print(f"unexpected y_orig shape: {y_orig.shape}")

        if y_feat_mix.shape != (12,512,768):
            print(f"unexpected y_feat_mix shape: {y_feat_mix.shape}")

        return y_orig,y_feat_mix ### y_orig for EAT loss, y_feat_mix for MixLoss



    # average top-k layers output from teacher model
    def make_targets(self, y, num_layers):

        with torch.no_grad():
            target_layer_results = y[-num_layers:]
            # print("type(target_layer_results) ",type(target_layer_results))
            if type(target_layer_results) is not list: ## when num_layers = 1
                target_layer_results = [target_layer_results]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BTC -> BCT
                ]
                permuted = True
            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]
            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]
            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]
            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]

        y = target_layer_results[0].float()
        for tl in target_layer_results[1:]:
            y.add_(tl.float())
        y = y.div_(len(target_layer_results))

        if self.cfg.layer_norm_targets:
            y = F.layer_norm(y, y.shape[-1:])

        if self.cfg.instance_norm_targets:
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)
        
        return y

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y**2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs**2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def extract_features(
        self, source, mode=None, padding_mask=None, mask=False, remove_extra_tokens=True
    ):
        res = self.forward(
            source,
            mode=mode,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            remove_extra_tokens=remove_extra_tokens,
        )
        return res

    def remove_pretraining_modules(self, modality=None, keep_decoder=False):
        self.ema = None
        self.cfg.clone_batch = 1
        self.recon_proj = None        

        if not keep_decoder:
            self.shared_decoder = None

        modality = modality.lower() if modality is not None else None
        for k in list(self.modality_encoders.keys()):
            if modality is not None and k.lower() != modality:
                del self.modality_encoders[k]
            else:
                self.modality_encoders[k].remove_pretraining_modules(
                    keep_decoder=keep_decoder
                )
                if not keep_decoder:
                    self.modality_encoders[k].decoder = None
