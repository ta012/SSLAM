#!/bin/bash


batch_size=48

echo "batch_size: $batch_size"

num_gpus=1


main_dir=path/to/fairseq/
echo "main_dir: $main_dir"



eat_dirname=SPASS_Finetune_LinearEval/SSLAM_Ft_SPASS/


required_num_epochs=50
ft_lr=0.00005

pretrained_weight=/path/to/SSLAM_Stage2/experiments/SSLAM_Stage2/checkpoint_last.pt

date_time_str=$(date +"%d_%m_%H_%M_%S")


### ["MARKET","PARK","SQUARE","STREET","WATERFRONT"]
for task_data in MARKET PARK SQUARE STREET WATERFRONT; do

    exp_dir_name=${date_time_str}_${task_data}
    echo "exp_dir_name name: $exp_dir_name"

    data_manifest_dir=/path/to/SPASS_Create_Manifests/manifest_SPASS_${task_data}/ 
    label_descriptors_csv=${data_manifest_dir}/label_descriptors.csv

    num_classes=$(wc -l < ${label_descriptors_csv})
    echo "num_classes: $num_classes"

    ### get max update
    train_lbl=${data_manifest_dir}/train.lbl
    echo "train_lbl: $train_lbl"
    num_train_samples=$(wc -l < ${train_lbl})
    echo "num_train_samples: $num_train_samples"
    max_update=$((num_train_samples*required_num_epochs/batch_size))
    warmup_update=$((max_update/10))
    echo "max_update: $max_update"
    echo "warmup_update: $warmup_update"


    ## assert data_manifest_dir exists
    if [ ! -d ${data_manifest_dir} ]; then
        echo "data_manifest_dir does not exist: ${data_manifest_dir}"
        exit 1
    fi

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
        task.data=${data_manifest_dir} \
        task.target_length=1024 \
        task.roll_aug=true \
        optimization.lr=[$ft_lr] \
        optimizer.groups.default.lr_float=$ft_lr \
        optimization.max_update=${max_update} \
        optimizer.groups.default.lr_scheduler.warmup_updates=${warmup_update} \
        model.model_path=${pretrained_weight} \
        model.num_classes=${num_classes} \
        model.mixup=0.8 \
        model.mask_ratio=0.2 \
        model.prediction_mode=PredictionMode.CLS_TOKEN > ${main_dir}/${eat_dirname}/experiments/${exp_dir_name}.log
done
