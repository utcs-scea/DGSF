import os
os.environ['FRAMEWORK'] = 'onnxrt'
import cv2
import numpy as np
from detect import RetinaFace, post_process, prepare_detection
from detect_preprocess import preprocess_img
from arcface_postprocess import get_aligned_face


def get_image_list(image_list):
    images = []
    with open(image_list, "r") as fin:
        for s in fin:
            image_name = s.strip()
            images.append(image_name)
    return images


def get_image_batch(img_list, dataset_path):
    padded_img_list = []
    rewrite_list = []
    for img_name in img_list:
        img_raw = cv2.imread(os.path.join(dataset_path, img_name))
        img_padded, rewrite = preprocess_img(img_raw)
        padded_img_list.append(img_padded)
        rewrite_list.append(rewrite)
    padded_img_list = np.array(padded_img_list)
    return padded_img_list, rewrite_list


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dmodel-path', required=True,
                        help="detection model path")
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--image-list', required=True)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()
    image_list = get_image_list(args.image_list)
    bs = 16
    model = RetinaFace(args.dmodel_path, "resnet50", bs, 1, "gpu")
    scale, scale1, priors = prepare_detection(model.cfg)
    length = len(image_list)
    for i in range(0, length, bs):
        sub_list = image_list[i:i+bs]
        padded_img_list, resize_list = get_image_batch(sub_list, args.dataset_path)
        loc_b, conf_b, landmarks_b, _ = \
            model.inference_batch(padded_img_list)
        pimg_list_len = len(padded_img_list)
        for j in range(pimg_list_len):
            dets_ = post_process(loc_b[j:j+1, :, :],
                                 conf_b[j:j+1, :, :], landmarks_b[j:j+1, :, :], model,
                                 priors, scale, scale1, resize_list[j])
            iname = os.path.join(args.dataset_path, sub_list[j])
            img_raw = cv2.imread(iname)
            dets = dets_[dets_[:, 4] >= model.vis_thres]
            aligned_list, _, _ = get_aligned_face(dets, img_raw)
            if aligned_list is None:
                print("no face in {}".format(iname))
            else:
                if len(aligned_list) != 1:
                    print("expecting each image only contains one face. {} is not".format(iname))
                dst = os.path.join(args.output_path, sub_list[j])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                np.save(dst, aligned_list[0])


if __name__ == '__main__':
    main()
