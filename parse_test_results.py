import argparse
import os
import shutil
import pandas as pd
import cv2
from tqdm import tqdm

from multiprocessing.pool import ThreadPool

Id2ClassName = {
    0: 'GRAFFITI',
    1: 'FADED_SIGNAGE',
    2: 'POTHOLES',
    3: 'GARBAGE',
    4: 'CONSTRUCTION_ROAD',
    5: 'BROKEN_SIGNAGE',
    6: 'BAD_STREETLIGHT',
    7: 'BAD_BILLBOARD',
    8: 'SAND_ON_ROAD',
    9: 'CLUTTER_SIDEWALK',
    10: 'UNKEPT_FACADE',
}

def create_submission_file(rows, submission_df):
    def preprocess(row):
        image_path, labels_path = row
        
        img = cv2.imread(image_path)
        img_shape = img.shape
        
        if os.path.exists(labels_path):
            label_df = pd.read_csv(labels_path, sep=" ", header=None, names=["class", "center_x", "center_y", "bbox_width", "bbox_height", "bbox_conf"])
            label_df.sort_values('bbox_conf')

            for _, row in label_df.iterrows():
                class_id = float(row["class"])
                center_x = row["center_x"]
                center_y = row["center_y"]
                bbox_width = row["bbox_width"]
                bbox_height = row["bbox_height"]

                xmin = center_x - bbox_width / 2
                xmax = center_x + bbox_width / 2
                ymin = center_y - bbox_height / 2
                ymax = center_y + bbox_height / 2

                xmin = max(xmin, 0)
                xmax = max(xmax, 0)
                ymin = max(ymin, 0)
                ymax = max(ymax, 0)

                xmin = min(xmin, 1)
                xmax = min(xmax, 1)
                ymin = min(ymin, 1)
                ymax = min(ymax, 1)

                xmin = int(xmin * img_shape[1] / 2)
                xmax = int(xmax * img_shape[1] / 2)
                ymin = int(ymin * img_shape[0] / 2)
                ymax = int(ymax * img_shape[0] / 2)

                submission_df.loc[len(submission_df.index)] = [class_id, os.path.basename(image_path), Id2ClassName[int(class_id)], float(xmax), float(xmin), float(ymax), float(ymin)]
        else:
            submission_df.loc[len(submission_df.index)] = [4.0, os.path.basename(image_path), 'CONSTRUCTION_ROAD', 0.0, 0.0, 0.0, 0.0]
        return
    
    for row in tqdm(rows):
        preprocess(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True, help="path to theme1 dataset")
    parser.add_argument("--test-dir", type=str, required=True, help="path to test output")
    parser.add_argument("--output-dir", type=str, required=True, help="path to output dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    test_images = os.path.join(args.dataset_dir, "test.csv")
    df = pd.read_csv(test_images)
    all_test_images = df["image_path"].unique()
    all_test_images = [(os.path.join(args.dataset_dir, "images", image_path), os.path.join(args.test_dir, image_path.replace('.jpg', '.txt'))) for image_path in all_test_images]

    submission_df = pd.DataFrame(columns=["class", "image_path", "name", "xmax", "xmin", "ymax", "ymin"])

    create_submission_file(
        all_test_images, 
        submission_df
    )
    submission_df.to_csv(os.path.join(args.output_dir, "submission.csv"), index=False)