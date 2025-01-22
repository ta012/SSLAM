#!/bin/bash

###########
batch_size=96

########

num_gpus=1

main_dir=/path/to/SSLAM/
echo "main_dir: $main_dir"
sslam_dirname=SSLAM_Inference

data_manifest_dir=$main_dir/data_manifests/manifest_as20k/ ## using  AudioSet eval set ## 
label_file=${data_manifest_dir}/label_descriptors.csv


### finetuned ckpt (AS2M_Ft)
finetuned_model_path=/path/to/SSLAM_model_weights/SSLAM_AS2M_Finetuned/checkpoint_best.pt


model_dir=${main_dir}/${sslam_dirname}/

/path/to/sslam_eval_env/bin/python ${main_dir}/${sslam_dirname}/evaluation/eval.py \
    --label_file=${label_file} \
    --model_dir=${model_dir} \
    --eval_dir=${data_manifest_dir} \
    --checkpoint_dir=${finetuned_model_path} \
    --target_length=1024 \
    --batch_size=${batch_size}
