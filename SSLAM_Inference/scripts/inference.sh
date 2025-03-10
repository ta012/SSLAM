#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# For optimal performance, 1024 is recommended for 10-second audio clips. (128 for 1-second)
# However, you should adjust the target_length parameter based on the duration and characteristics of your specific audio inputs.
# EAT-finetuned could make inference well even given truncated audio clips.


main_dir=/path/to/SSLAM/

echo "main_dir: $main_dir"
sslam_dirname=SSLAM_Inference

label_file=${main_dir}/${sslam_dirname}/inference/labels.csv


### finetuned ckpt (AS2M_Ft)
finetuned_model_path=/path/to/SSLAM_AS2M_Finetuned/checkpoint_best.pt


model_dir=${main_dir}/${sslam_dirname}/

sample_wav=${main_dir}/${sslam_dirname}/inference/test.wav


/path/to/sslam_eval_minimal_env/bin/python ${main_dir}/${sslam_dirname}/inference/inference.py  \
    --source_file=$sample_wav \
    --label_file=$label_file \
    --model_dir=$model_dir \
    --checkpoint_dir=$finetuned_model_path \
    --target_length=1024 \
    --top_k_prediction=12 \
