#!/bin/bash


batch_size=48
max_update=40000
warmup_update=4000
echo "batch_size: $batch_size"

num_gpus=1

mnt_vol=/mnt/fast/nobackup/users/ta01123/
main_dir=${mnt_vol}/eat_apr30/fairseq/
echo "main_dir: $main_dir"
eat_dirname=EAT_FtCkptMixRegionExtraMixLCls1Layer2S2TDrpInpTBothNoMixCls
exp_dir_name=$(date +"%d_%m_%H_%M_%S")
echo "exp_dir_name name: $exp_dir_name"

# optimizer.groups.default.lr_scheduler.min_lr=0.00005 \

pretrained_weight=/mnt/fast/nobackup/users/ta01123/eat_apr30/fairseq/EAT_CkptMixRegionExtraMixLCls1Layer2S2TDrpInpTBothNoMixCls/experiments/EAT_CkptMixRegionExtraMixLCls1Layer2S2TDrpInpTBothNoMixCls/checkpoint_last.pt


/mnt/fast/nobackup/users/ta01123/eat_apr30/fairseqvenv/bin/python ${main_dir}/fairseq_cli/hydra_train.py -m \
    --config-dir ${main_dir}/${eat_dirname}/config \
    --config-name finetuning  \
    common.user_dir=${eat_dirname} \
    hydra.sweep.dir=${main_dir}/${eat_dirname}/experiments/${exp_dir_name}_multirun/ \
    checkpoint.save_dir=/mnt/fast/nobackup/scratch4weeks/ta01123/to_del_eat/${exp_dir_name}/ \
    checkpoint.restore_file=None \
    checkpoint.save_interval=500000 \
    checkpoint.best_checkpoint_metric=mAP \
    checkpoint.no_save=true \
    checkpoint.no_last_checkpoints=true \
    checkpoint.no_save_optimizer_state=true \
    dataset.batch_size=${batch_size} \
    task.data=${mnt_vol}/eat_data_manifest/manifest_as20k/ \
    task.target_length=1024 \
    task.roll_aug=true \
    optimization.max_update=${max_update} \
    optimizer.groups.default.lr_scheduler.warmup_updates=${warmup_update} \
    model.model_path=${pretrained_weight} \
    model.num_classes=527 \
    model.mixup=0.8 \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN > ${main_dir}/${eat_dirname}/experiments/${exp_dir_name}.log
