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

source_path=${1:-"/data/Theme1_dataset_yolo/images/test"}
trained_model=${2:-"/output/yolov6_train_output/exp/weights/best_ckpt.pt"}
output_path=${3:-"/output/yolov6_evaluation_output"}

# Remove if output path exists
mkdir -p $output_path
rm -r $output_path/* || true

cd yolov6
export PYTHONPATH=$PYTHONPATH:$(pwd)/yolov6 && \
python tools/infer.py \
    --source $source_path \
    --yaml yaml_configs/train_data.yaml \
    --weights $trained_model \
    --conf-thres 0.4 \
    --save-txt \
    --save-dir $output_path \
    --device 0 2>&1 | tee $output_path/test_log.txt

cd ..
python parse_test_results.py \
            --dataset-dir /data/Theme1_dataset \
            --test-dir $output_path/test \
            --output-dir $output_path