import argparse
import os
import cv2
import numpy as np
from detect_preprocess import preprocess_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', help="face image data path",
                        required=True)
    parser.add_argument('--image-list', help="image list", required=True)
    parser.add_argument('--out-dir', default="preprocessed", required=True)
    args = parser.parse_args()

    with open(args.image_list, 'r') as f:
        for s in f:
            image_name = s.strip()
            src = os.path.join(args.data_path, image_name)
            dst = os.path.join(args.out_dir, image_name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            img_raw = cv2.imread(src)
            processed_img, _ = preprocess_img(img_raw)
            np.save(dst, processed_img)


if __name__ == '__main__':
    main()
