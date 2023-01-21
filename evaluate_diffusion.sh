#!/bin/bash
set -ex

# Run from $WORKDIR

# Note: 
# please mount /data to a directory containing unzipped Theme1 dataset on your host machine

# ├── Theme1_dataset
# │   ├── images/
# │   │   ├── 0a1ea4614a9df912eeb8d1b40bffee74.jpg
# │   │   ├── 0a2bc0dc2371794509f4b776aff0dd88.jpg
# │   │   ├── ...
# │   │   └── 0a82e45ed11fb9ef1620a0b40cd9f6d8.jpg
# │   ├── sample_submission.csv
# │   ├── test.csv
# |   └── train.csv

trained_model=${1:-"/output/diffusion_train_output/model_final.pth"}
output_path=${1:-"/output/diffusion_evaluation_output"}

# Remove if output path exists
mkdir -p $output_path
rm -r $output_path/* || true

cd DiffusionDet
export PYTHONPATH=$PYTHONPATH:$(pwd)/DiffusionDet && \
python demo.py --config-file configs/diffdet.coco.res50.finetune.yaml \
        --input /data/Theme1_dataset_yolo/images/test/*.jpg \
        --output $output_path \
        --confidence-threshold 0.5 \
        --opts MODEL.WEIGHTS $trained_model 2>&1 | tee $output_path/test_log.txt