#!/bin/bash

## for 24gb
batch_size=12

num_gpus=4

max_updates=200000
base_lr=0.00005
warmup=25000

###
main_dir=/path/to/fairseq/
echo "main_dir: $main_dir"
exp_dir_name=SSLAM_Stage2
sslam_dir=SSLAM_Stage2

log_file=$(date +"%d_%m_%H_%M_%S")
echo "Folder name: $exp_dir_name"
mkdir -p ${main_dir}/${sslam_dir}/experiments/${exp_dir_name}

# Setting the Hydra output directories to absolute paths
clone_batch=8 ## (after concat orig and mix effective is same as 16)

stage1_checkpoint_path=/path/to/eat_ckpt/EAT-base_epoch10_pt.pt  ## Stage 1 pre-training is identical to EAT, both using 10 epochs and achieving the same performance.


manifest_as2mfull_pretrain_path=/path/to/data_manifests/manifest_as2mfull_pre/ 

/path/to/fairseqvenv/bin/python ${main_dir}/fairseq_cli/hydra_train.py -m \
    --config-dir ${main_dir}/${sslam_dir}/config \
    --config-name pretraining_AS2M \
    common.user_dir=${sslam_dir} \
    hydra.sweep.dir=${main_dir}/${sslam_dir}/experiments/${exp_dir_name}_multirun \
    optimization.max_update=${max_updates} \
    optimization.lr=[${base_lr}] \
    optimizer.groups.default.lr_float=${base_lr} \
    optimizer.groups.default.lr_scheduler.warmup_updates=${warmup} \
    checkpoint.save_dir=${main_dir}/${sslam_dir}/experiments/${exp_dir_name} \
    checkpoint.continue_once=$stage1_checkpoint_path \
    checkpoint.reset_dataloader=True \
    checkpoint.reset_lr_scheduler=True \
    checkpoint.reset_meters=True \
    checkpoint.reset_optimizer=True \
    checkpoint.save_interval_updates=40262 \
    checkpoint.keep_interval_updates=8 \
    distributed_training.distributed_world_size=${num_gpus} \
    dataset.batch_size=${batch_size} \
    task.data=$manifest_as2mfull_pretrain_path \
    model.clone_batch=${clone_batch} \
    task.h5_format=False  > ${main_dir}/${sslam_dir}/experiments/${exp_dir_name}/${log_file}.log


    
