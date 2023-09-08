import cv2
import numpy as np


def calculate_resize(img_shape):
    target_size = 640
    max_size = 640
    im_size_min = np.min(img_shape[0:2])
    im_size_max = np.max(img_shape[0:2])
    resize = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    return resize


def preprocess_img(img_raw):
    img_shape = img_raw.shape
    resize = calculate_resize(img_shape)
    img = np.array(img_raw, dtype=np.int16)

    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize,
                         interpolation=cv2.INTER_LINEAR)

    img -= np.array((104, 117, 123), dtype=np.int16)
    img = img.transpose((2, 0, 1))
    img_padded = np.zeros((3, 640, 640), dtype=np.int16)
    img_padded[:img.shape[0], :min(img.shape[1], 640),
               :min(img.shape[2], 640)] = img
    return img_padded, resize
