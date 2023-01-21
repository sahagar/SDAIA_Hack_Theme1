import argparse
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

from multiprocessing.pool import ThreadPool

def copy_file(src_file, dst_file):
    shutil.copyfile(src_file, dst_file)
    return

def img_normalize(xyxy, img_shape):
    x1, y1, x2, y2 = xyxy
    x1 = x1 / img_shape[1]
    x2 = x2 / img_shape[1]
    y1 = y1 / img_shape[0]
    y2 = y2 / img_shape[0]
    return [x1, y1, x2, y2]

def preprocess_dataset(rows, output_images_dir, output_labels_dir=None):
    def preprocess(row):
        image_path, labels = row
        copy_file(image_path, os.path.join(output_images_dir, os.path.basename(image_path)))
        img = cv2.imread(image_path)
        img_shape = img.shape

        if output_labels_dir is not None and len(labels) > 0:
            with open(os.path.join(output_labels_dir, os.path.basename(image_path).replace(".jpg", ".txt")), "w") as f:
                for label in labels:
                    class_id = int(label[0])
                    xmax, xmin, ymax, ymin = 2*float(label[-4]), 2*float(label[-3]), 2*float(label[-2]), 2*float(label[-1])
                    
                    xmax = min(xmax, img_shape[1])
                    xmin = min(xmin, img_shape[1])
                    ymax = min(ymax, img_shape[0])
                    ymin = min(ymin, img_shape[0])

                    xmax = max(xmax, 0)
                    xmin = max(xmin, 0)
                    ymax = max(ymax, 0)
                    ymin = max(ymin, 0)
                    
                    xmax, ymax, xmin, ymin = img_normalize([xmax, ymax, xmin, ymin], img_shape)

                    center_x = (xmax + xmin) / 2
                    center_y = (ymax + ymin) / 2
                    bbox_width = abs(xmax - xmin)
                    bbox_height = abs(ymax - ymin)
                    f.write(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}\n")
                f.close()
        return
    
    with ThreadPool(processes=16) as p:
        p.map(preprocess, rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True, help="path to theme1 dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="path to output dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images", "valid"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "labels", "valid"), exist_ok=True)
    
    label_file = os.path.join(args.dataset_dir, "train.csv")
    df = pd.read_csv(label_file)

    classes = df["class"].unique()
    train_images, valid_images = [], []
    for class_id in classes:
        class_images = list(df[df["class"] == class_id]["image_path"].unique())
        try:
            class_train_images, class_valid_images = train_test_split(class_images, test_size=0.04, random_state=42)
        except Exception as e:
            print(f"For class id={class_id} there are only {len(class_images)} images.")
            print(e)
            class_train_images, class_valid_images = class_images, []
        train_images += class_train_images
        valid_images += class_valid_images

    train_images = [(os.path.join(args.dataset_dir, "images", image_path), df[df["image_path"] == image_path][['class', 'name', 'xmax', 'xmin', 'ymax', 'ymin']].values) for image_path in train_images]
    valid_images = [(os.path.join(args.dataset_dir, "images", image_path), df[df["image_path"] == image_path][['class', 'name', 'xmax', 'xmin', 'ymax', 'ymin']].values) for image_path in valid_images]

    print("Preprocessing train dataset...")
    print("Number of train images: ", len(train_images))
    print("Output images dir: ", os.path.join(args.output_dir, "images", "train"))
    print("Output labels dir: ", os.path.join(args.output_dir, "labels", "train"))
    preprocess_dataset(
        train_images, 
        output_images_dir=os.path.join(args.output_dir, "images", "train"),
        output_labels_dir=os.path.join(args.output_dir, "labels", "train")
    )

    print("Preprocessing valid dataset...")
    print("Number of valid images: ", len(valid_images))
    print("Output images dir: ", os.path.join(args.output_dir, "images", "valid"))
    print("Output labels dir: ", os.path.join(args.output_dir, "labels", "valid"))
    preprocess_dataset(
        valid_images,
        output_images_dir=os.path.join(args.output_dir, "images", "valid"),
        output_labels_dir=os.path.join(args.output_dir, "labels", "valid")
    )

    test_images = os.path.join(args.dataset_dir, "test.csv")
    df = pd.read_csv(test_images)
    all_test_images = df["image_path"].unique()
    all_test_images = [(os.path.join(args.dataset_dir, "images", image_path), []) for image_path in all_test_images]

    print("Preprocessing test dataset...")
    print("Number of test images: ", len(all_test_images))
    print("Output images dir: ", os.path.join(args.output_dir, "images", "test"))
    preprocess_dataset(
        all_test_images, 
        output_images_dir=os.path.join(args.output_dir, "images", "test"),
        output_labels_dir=None
    )
    