import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import fairseq
import torchaudio
from sklearn import metrics as sklearn_metrics
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import json

def get_parser():
    parser = argparse.ArgumentParser(description="Use fine-tuned EAT for inference in Audioset-eval")
    parser.add_argument('--eval_dir', type=str, help='Directory with eval.lbl and eval.tsv files', required=True)
    parser.add_argument('--label_file', help='Location of label files', required=True)
    parser.add_argument('--model_dir', type=str, help='Pretrained model directory', required=True)
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint for fine-tuned model', required=True)
    parser.add_argument('--target_length', type=int, help='Target length of Mel spectrogram in time dimension', required=True)
    parser.add_argument('--norm_mean', type=float, help='Mean value for normalization', default=-4.268)
    parser.add_argument('--norm_std', type=float, help='Standard deviation for normalization', default=4.569)
    parser.add_argument('--device', type=str, help='Device to run the model on (cpu or cuda)', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, help='Inference batch size in dataloader', default=16)
    parser.add_argument('--ap_log_path', type=str, default='EAT/ap_log.txt', help='Path to save the AP values log')
    return parser


def build_dictionary(label_path):
    vocab = {}
    with open(label_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            index = int(row[0])
            label = row[1]
            vocab[label] = index
    return vocab

def build_dictionary_2(label_path):
    vocab = {}
    with open(label_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            index = int(row[0])
            label = row[2]
            vocab[index] = label
    return vocab


# def load_audio_labels(eval_dir):
#     audio_paths = []
#     labels = {}
#     root_dir = ''
    
#     with open(Path(eval_dir) / 'eval.tsv', 'r') as tsv_file:
#         for line in tsv_file:
#             parts = line.strip().split('\t')
#             if len(parts) == 2:
#                 audio_paths.append((Path(root_dir) / parts[0], int(parts[1])))
#             else:
#                 root_dir = parts[0]
#     with open(Path(eval_dir) / 'eval.lbl', 'r') as lbl_file:
#         for line in lbl_file:
#             parts = line.strip().split('\t')
#             if len(parts) == 2:
#                 key = parts[0] + '.wav'
#                 labels[key] = parts[1].split(',')
#     return audio_paths, labels

def load_audio_labels_apr25(dataset_json_file):
    with open(dataset_json_file, 'r') as fp:
        data_json = json.load(fp)

    data = data_json['data']
    audio_paths = []
    labels = dict()
    for i in range(len(data)):
        d = data[i]
        wav_name = d['wav'].split('/')[-1]
        a_path = d['wav']

        audio_paths.append((Path(a_path),'SR why used?'))
        # print(audio_paths[-1])
        wav_name = d['wav'].split('/')[-1]
        labels[wav_name] = d['labels'].split(',')
    
    return audio_paths, labels


def calculate_map(output, target, inx_lbl):
    classes_num = target.shape[-1]
    ap_values = {}
    for k in range(classes_num):
        avg_precision = sklearn_metrics.average_precision_score(target[:, k], output[:, k], average=None)
        ap_values[inx_lbl[k]] = avg_precision
    mean_ap = np.nanmean(list(ap_values.values()))
    return mean_ap, ap_values



@dataclass
class UserDirModule:
    user_dir: str



class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels, vocab, target_length, norm_mean, norm_std, device):
        self.audio_paths = audio_paths
        self.labels = labels
        self.vocab = vocab
        self.target_length = target_length
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.device = device
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path, _ = self.audio_paths[idx]
        label = self.labels[audio_path.name]
        wav, sr = sf.read(audio_path)
        source = torch.from_numpy(wav).float().to(self.device)
        if sr != 16e3:
            source = torchaudio.functional.resample(source, orig_freq=sr, new_freq=16000).float()
        source = source - source.mean()
        source = source.unsqueeze(dim=0)
        source = torchaudio.compliance.kaldi.fbank(source, htk_compat=True, sample_frequency=16000, use_energy=False,
                                                    window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10).unsqueeze(dim=0)
        n_frames = source.shape[1]
        diff = self.target_length - n_frames
        if diff > 0:
            source = F.pad(source, (0, 0, 0, diff))
        elif diff < 0:
            source = source[:,:self.target_length,:]
        source = (source - self.norm_mean) / (self.norm_std * 2)
        target = torch.zeros(len(self.vocab)).to(self.device)
        for lbl in label:
            if lbl in self.vocab:
                idx = self.vocab[lbl]
                target[idx] = 1
        return source, target



def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    
    device = torch.device(args.device)

    vocab = build_dictionary(args.label_file)
    inx_lbl = build_dictionary_2(args.label_file)
    eval_json = '/mnt/fast/nobackup/users/ta01123/audioset/eval_data_weka.json'
    audio_paths, labels = load_audio_labels_apr25(eval_json)
    
    # Load the model
    model_path = UserDirModule(args.model_dir)
    fairseq.utils.import_user_module(model_path)
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([args.checkpoint_dir])
    model = model[0].eval().to(device)
    
    dataset = AudioDataset(audio_paths, labels, vocab, args.target_length, args.norm_mean, args.norm_std, device)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        
    outputs = []
    targets = []
    with torch.no_grad():
        for sources, target in tqdm(data_loader, desc="Testing"):
            pred = model(sources)
            pred = torch.sigmoid(pred)
            outputs.append(pred)
            targets.append(target)
            
    outputs_tensor = torch.cat(outputs, dim=0)
    targets_tensor = torch.cat(targets, dim=0)
    mAP, ap_values  = calculate_map(outputs_tensor.cpu().numpy(), targets_tensor.cpu().numpy(),inx_lbl)
    print(f"The fine-tuned model's performance on Audioset-eval with {len(dataset)} audio clips is: {mAP:.4f}")

    with open(args.ap_log_path, 'w') as log_file:
        log_file.write(f"{'Class':<50s} AP\n")
        for k, ap in ap_values.items():
            log_file.write(f"{k:<50s} {ap:.4f}\n")
"""
/mnt2/data/AudioSet/eval_segments/Y7ho9Ro2ElOA.wav
(base) tony@tony-TUF-Gaming-FX505GD-FX505GD:/mnt2/eat_apr24/fairseq$ bash EAT/scripts/eval.sh 0
Namespace(ap_log_path='EAT/ap_log.txt', batch_size=32, checkpoint_dir='/mnt2/eat_apr24/eat_ckpts/EAT-base_epoch30_ft.pt', device='cuda', eval_dir='None', label_file='EAT/inference/labels.csv', model_dir='EAT', norm_mean=-4.268, norm_std=4.569, target_length=1024)
2024-04-25 20:13:21 | INFO | EAT.models.EAT_pretraining | making target model
/mnt2/eat_apr24/fairseq_env/lib/python3.8/site-packages/torch/nn/modules/module.py:1879: UserWarning: Positional args are being deprecated, use kwargs instead. Refer to https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.state_dict for details.
  warnings.warn(
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 591/591 [47:20<00:00,  4.81s/it]
The fine-tuned model's performance on Audioset-eval with 18886 audio clips is: 0.4885

"""
if __name__ == '__main__':
    main()
