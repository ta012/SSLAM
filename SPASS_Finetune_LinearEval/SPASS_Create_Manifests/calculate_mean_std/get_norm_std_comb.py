#!/mnt/fast/nobackup/users/ta01123/ibot_env/bin/python

# -*- coding: utf-8 -*-
# @Time    : 8/4/21 4:30 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_norm_stats.py

# this is a sample code of how to get normalization stats for input spectrogram

import torch
import numpy as np
import os
import dataloader
import pandas as pd
# set skip_norm as True only when you are computing the normalization stats
audio_conf = {'num_mel_bins': 128, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'skip_norm': True, 'mode': 'train', 'dataset': 'audioset'}

main_dir="/mnt/fast/nobackup/scratch4weeks/ta01123/SPASS/"
train_l_all = []


for subset in  ["MARKET","PARK","SQUARE","STREET","WATERFRONT"]:
    train_tsv = f"manifest_SPASS_{subset}/train.tsv"
    train_tsv = '/mnt/fast/nobackup/users/ta01123/eat_apr30/fairseq/SPASS/SPASS_Create_Manifests/' + train_tsv
    wavs_l = pd.read_csv(train_tsv, sep='\t',header=None, skiprows=1)[0].tolist()
    print(f"len(wavs_l) {len(wavs_l)}, wavs_l[:2] {wavs_l[:2]}")
    train_l_all.extend(wavs_l)
    
    

print(f"len(train_l_all) {len(train_l_all)}")
train_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(None, label_csv=None,
                                audio_conf=audio_conf,main_dir=main_dir,train_l=train_l_all), batch_size=1000, shuffle=False, num_workers=8, pin_memory=True)
mean=[]
std=[]
for i, (audio_input, labels) in enumerate(train_loader):
    cur_mean = torch.mean(audio_input)
    cur_std = torch.std(audio_input)
    mean.append(cur_mean)
    std.append(cur_std)
    print(cur_mean, cur_std)
print(np.mean(mean), np.mean(std))
