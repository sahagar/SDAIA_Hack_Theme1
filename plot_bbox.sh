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

images_dir=${1:-"/data/Theme1_dataset/images"}
label_file=${2:-"/data/Theme1_dataset/train.csv"}
output_dir=${3:-"/data/Theme1_dataset_plotted"}

python plot_bbox.py \
        --image-dir $images_dir \
        --label-file $label_file \
        --output-dir $output_dir