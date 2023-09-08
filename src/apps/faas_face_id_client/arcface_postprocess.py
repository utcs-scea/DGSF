import numpy as np
from timeit import default_timer as timer
from skimage import transform as trans
import cv2


def get_face(img, bbox=None, landmark=None, **kwargs):
    """
    get the face in bbox and align using landmark
    """
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    # Assert input shape
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
        assert image_size[0] == 112
        assert image_size[0] == 112 or image_size[1] == 96

    # Do alignment using landmark points
    if landmark is not None:
        assert len(image_size) == 2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        lmk = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(lmk, src)
        M = tform.params[0:2, :]
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]),
                                borderValue=0.0)
        return warped

    # If no landmark points available, do alignment using bounding box.
    # If no bounding box available use center crop
    if M is None:
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret


def align_one_face(det, img):
    if len(det) == 0:
        return None, None
    bbox = det[:4]
    points = det[5:15].astype(int)
    points = points.reshape((5, 2))
    nimg = get_face(img, bbox, points, image_size='112,112')
    aligned = np.transpose(nimg, (2, 0, 1))
    return bbox, aligned


def get_aligned_face(dets, img):
    bbox_list = []
    aligned_list = []
    get_face_t = 0
    if len(dets.shape) == 1:
        start = timer()
        bbox, aligned = align_one_face(dets, img)
        get_face_t += timer() - start
        if bbox is None or aligned is None:
            return aligned_list, bbox_list, get_face_t
        bbox_list.append(bbox)
        aligned_list = np.expand_dims(aligned, axis=0)
        return aligned_list, bbox_list, get_face_t

    for det in dets:
        if len(det) == 0:
            continue
        start = timer()
        bbox, aligned = align_one_face(det, img)
        get_face_t += timer() - start
        bbox_list.append(bbox)
        aligned_list.append(np.expand_dims(aligned, axis=0))
    if len(aligned_list) != 0:
        aligned_list = np.concatenate(aligned_list, axis=0)
    else:
        aligned_list = None
    return aligned_list, bbox_list, get_face_t
