#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


main_dir=/path/to/SSLAM/
echo "main_dir: $main_dir"
sslam_dirname=SSLAM_Inference

data_manifest_dir=$main_dir/data_manifests/manifest_as20k/


### finetuned ckpt (AS2M_Ft/AS20K_Ft)
pretrained_model_path=/path/to/SSLAM_model_weights/SSLAM_Pretrained/checkpoint_last.pt
# finetuned_model_path=/path/to/SSLAM_model_weights/SSLAM_AS2M_Finetuned/checkpoint_best.pt


model_dir=${main_dir}/${sslam_dirname}/

sample_wav=${main_dir}/${sslam_dirname}/feature_extract/test.wav
output_npy=${main_dir}/${sslam_dirname}/feature_extract/test.npy



granularity='frame' ## frame: patch tokens/features, utterance: cls token, all: both patch/frame and cls tokens
mode='pretrain' ## pretrain: use this if you are using pre-trained checkpoint, finetune: use this if you are using finetuned checkpoint



/path/to/sslam_eval_minimal_env/bin/python ${main_dir}/${sslam_dirname}/feature_extract/feature_extract.py  \
    --source_file=$sample_wav \
    --target_file=$output_npy \
    --model_dir=${model_dir} \
    --checkpoint_dir=${pretrained_model_path}  \
    --granularity=$granularity \
    --target_length=1024 \
    --mode=$mode