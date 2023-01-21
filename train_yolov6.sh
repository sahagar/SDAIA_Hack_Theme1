#!/bin/bash
set -ex

# Run from $WORKDIR

# Note:
# This script assumes dataset is processed by preprocess_data.py
# train_data.yaml assumes the data is located in /data/Theme1_dataset_yolov6. Please change paths as needed.
# please mount /output to a directory on your host machine to save checkpoints and logs
# Trained on 1 A6000 GPU

export RANK=0
export MASTER_ADDR=localhost

output_path=${1:-"/output/yolov6_train_output"}
rm -r $output_path/* || true
mkdir -p $output_path

cd yolov6
export PYTHONPATH=$PYTHONPATH:$(pwd)/yolov6 && \
torchrun \
    --nnodes 1 --nproc_per_node 1 \
    --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=29510 \
    tools/train.py \
        --batch 64 \
        --epochs 100 \
        --conf configs/yolov6l6_finetune.py \
        --data yaml_configs/train_data.yaml \
        --output-dir $output_path \
        --device 0 2>&1 | tee $output_path/yolov6_train_log.txt