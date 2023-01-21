import argparse
import pandas as pd
import os
import cv2
from tqdm import tqdm

def generate_colors(i, bgr=False):
    hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
            '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
    palette = []
    for iter in hex:
        h = '#' + iter
        palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
    num = len(palette)
    color = palette[int(i) % num]
    return (color[2], color[1], color[0]) if bgr else color

def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)

# Multiply by 2 because the annotaions for Theme 1 assume image width to be 1920/2 = 960
def get_box(xmax, xmin, ymax, ymin):
    xmax, xmin, ymax, ymin = 2*xmax, 2*xmin, 2*ymax, 2*ymin
    return [xmin, ymin, xmax, ymax]

def plot_for_all_images(image_dir, label_file, output_dir):
    df = pd.read_csv(label_file)
    images = list(df['image_path'].unique())
    classes = list(df['name'].unique())

    for image in tqdm(images):
        df_image = df[df['image_path'] == image]
        values = df_image[['name', 'xmax', 'xmin', 'ymax', 'ymin']].values
        
        image_path = os.path.join(image_dir, image)
        img = cv2.imread(image_path)
        
        for value in values:
            class_name, xmax, xmin, ymax, ymin = value
            plot_box_and_label(img, max(round(sum(img.shape) / 2 * 0.003), 2), get_box(xmax, xmin, ymax, ymin), class_name, color=generate_colors(classes.index(class_name), True))
        
        save_path = os.path.join(output_dir, image)
        cv2.imwrite(save_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True, help="path to image")
    parser.add_argument("--label-file", type=str, required=True, help="path to label file")
    parser.add_argument("--output-dir", type=str, required=True, help="path to output dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plot_for_all_images(args.image_dir, args.label_file, args.output_dir)