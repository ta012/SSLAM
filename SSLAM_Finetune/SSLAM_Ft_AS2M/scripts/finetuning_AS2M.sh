#!/bin/bash

###########
batch_size=96

ft_lr=0.00005

max_update=400000  
warmup_update=40000 
########

num_gpus=1

main_dir=/path/to/fairseq/
echo "main_dir: $main_dir"
eat_dirname=SSLAM_Ft_AS2M

exp_dir_name=$(basename "$eat_dirname") ## take last part of eat_dirname

echo "exp_dir_name name: $exp_dir_name"

mkdir -p ${main_dir}/${eat_dirname}/experiments/${exp_dir_name}

d_m_hh_mm_ss=$(date +"%d_%m_%H_%M_%S")

log_file=${main_dir}/${eat_dirname}/experiments/${exp_dir_name}/${exp_dir_name}_${d_m_hh_mm_ss}_log.log

pretrained_weight=/path/to/SSLAM_Stage2/experiments/SSLAM_Stage2/checkpoint_last.pt


/path/to/fairseqvenv/bin/python ${main_dir}/fairseq_cli/hydra_train.py -m \
    --config-dir ${main_dir}/${eat_dirname}/config \
    --config-name finetuning  \
    common.user_dir=${eat_dirname} \
    hydra.sweep.dir=${main_dir}/${eat_dirname}/experiments/${exp_dir_name}_multirun/ \
    checkpoint.save_dir=${main_dir}/${eat_dirname}/experiments/${exp_dir_name}/ \
    checkpoint.restore_file=${main_dir}/${eat_dirname}/experiments/${exp_dir_name}/checkpoint_last.pt \
    checkpoint.best_checkpoint_metric=mAP \
    dataset.batch_size=${batch_size} \
    task.data=${main_dir}/eat_data_manifest/manifest_as2mfull_ft \
    task.h5_format=false \
    task.num_samples=200000 \
    task.AS2M_finetune=true \
    task.weights_file=${main_dir}/eat_data_manifest/manifest_as2mfull_ft/merged_unbal_bal_train_weka_weight.csv \
    task.target_length=1024 \
    task.roll_aug=true \
    optimization.lr=[$ft_lr] \
    optimizer.groups.default.lr_float=$ft_lr \
    optimization.max_update=${max_update} \
    optimizer.groups.default.lr_scheduler.warmup_updates=${warmup_update} \
    model.model_path=${pretrained_weight} \
    model.num_classes=527 \
    model.mixup=0.8 \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN > ${log_file}