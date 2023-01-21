#!/bin/bash
set -ex

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

datatset_path=${1:-"/data/Theme1_dataset"}
output_path=${2:-"/data/Theme1_dataset_yolo"}

python preprocess_data.py \
        --dataset-dir $datatset_path \
        --output-dir $output_path