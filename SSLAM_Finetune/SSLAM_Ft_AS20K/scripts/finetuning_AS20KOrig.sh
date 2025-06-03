#!/bin/bash


batch_size=48
max_update=40000
warmup_update=4000
echo "batch_size: $batch_size"

num_gpus=1

main_dir=/path/to/fairseq/
echo "main_dir: $main_dir"
eat_dirname=SSLAM_Ft_AS20K
exp_dir_name=$(date +"%d_%m_%H_%M_%S")
echo "exp_dir_name name: $exp_dir_name"

# optimizer.groups.default.lr_scheduler.min_lr=0.00005 \

pretrained_weight=/path/to/SSLAM_Stage2/experiments/SSLAM_Stage2/checkpoint_last.pt


/path/to/fairseqvenv/bin/python ${main_dir}/fairseq_cli/hydra_train.py -m \
    --config-dir ${main_dir}/${eat_dirname}/config \
    --config-name finetuning  \
    common.user_dir=${eat_dirname} \
    hydra.sweep.dir=${main_dir}/${eat_dirname}/experiments/${exp_dir_name}_multirun/ \
    checkpoint.save_dir=${main_dir}/${eat_dirname}/experiments/${exp_dir_name}/ \
    checkpoint.restore_file=None \
    checkpoint.save_interval=500000 \
    checkpoint.best_checkpoint_metric=mAP \
    checkpoint.no_save=true \
    checkpoint.no_last_checkpoints=true \
    checkpoint.no_save_optimizer_state=true \
    dataset.batch_size=${batch_size} \
    task.data=${main_dir}/eat_data_manifest/manifest_as20k/ \
    task.target_length=1024 \
    task.roll_aug=true \
    optimization.max_update=${max_update} \
    optimizer.groups.default.lr_scheduler.warmup_updates=${warmup_update} \
    model.model_path=${pretrained_weight} \
    model.num_classes=527 \
    model.mixup=0.8 \
    model.mask_ratio=0.2 \
    model.prediction_mode=PredictionMode.CLS_TOKEN > ${main_dir}/${eat_dirname}/experiments/${exp_dir_name}.log
