#!/bin/bash
set -ex

# Run from $WORKDIR

# Note:
# This script assumes dataset is processed by preprocess_data.py
# diffdet.swinbase.finetune.yaml assumes the data is located in /data/Theme1_dataset_yolov6. Please change paths as needed.
# please mount /output to a directory on your host machine to save checkpoints and logs
# Trained on 1 A6000 GPU

export RANK=0
export MASTER_ADDR=localhost

output_path=${1:-"/output/diffusion_train_output"}
rm -r $output_path/* || true
mkdir -p $output_path

cd DiffusionDet
export PYTHONPATH=$PYTHONPATH:$(pwd)/yolov6 && \
python train_net.py --num-gpus 1 \
    --config-file configs/diffdet.coco.res50.finetune.yaml \
    --data-dir /data/Theme1_dataset_yolo/images \
    SEED 42 \
    OUTPUT_DIR $output_path 2>&1 | tee $output_path/diffusion_train_log.txt