import os
from timeit import default_timer as timer
import cv2
import numpy as np


class Dataset:
    def __init__(self, data_path, image_list):
        self.image_list = []
        self.data_path = data_path
        with open(image_list, 'r') as f:
            for s in f:
                image_name = s.strip()
                self.image_list.append(os.path.join(data_path, image_name))

    def get_item(self, nr):
        dst = os.path.join(self.data_path, self.image_list[nr])
        img = cv2.imread(dst)
        return img

    def load_samples_and_resize(self, sample_ids):
        from detect_preprocess import preprocess_img
        padded_image_list = []
        resize_list = []
        dl_t = 0
        for sample_id in sample_ids:
            dl_s = timer()
            img_raw = self.get_item(sample_id)
            dl_e = timer()
            dl_t += dl_e - dl_s
            img_padded, resize = preprocess_img(img_raw)
            padded_image_list.append(img_padded)
            resize_list.append(resize)
        padded_image_list = np.array(padded_image_list)
        return padded_image_list, resize_list, dl_t


DS = Dataset('/data/', '/data/wider_val.txt')


class Runner:
    def __init__(self, model, ds, batch_size):
        from detect import prepare_detection
        self.ds = ds
        self.model = model
        self.batch_size = batch_size
        self.scale, self.scale1, self.priors = prepare_detection(model.cfg)

    def run_one_item(self, idx, padded_image_list, rewrite_list):
        from detect import post_process
        loc_b, conf_b, landmarks_b, net_time = \
            self.model.inference_batch(padded_image_list)
        length = len(padded_image_list)
        return_list = []
        post_s = timer()
        for i in range(length):
            dets_ = post_process(loc_b[i:i+1, :, :], conf_b[i:i+1, :, :],
                                 landmarks_b[i:i+1, :, :],
                                 self.model, self.priors,
                                 self.scale, self.scale1, rewrite_list[i])
            dets = dets_[dets_[:, 4] >= self.model.vis_thres]
            dets_list = dets.tolist()
            return_list.append((idx[i], dets_list))
        post_e = timer()
        post_t = post_e - post_s
        return return_list, net_time, post_t

    def run_samples(self, query_samples):
        idx = [q['idx'] for q in query_samples]
        if len(query_samples) < self.batch_size:
            prep_s = timer()
            padded_image_list, rewrite_list, dl = \
                self.ds.load_samples_and_resize(idx)
            prep_e = timer()
            prep_t = prep_e - prep_s - dl
            ret_list, net_t, post_t = self.run_one_item(idx, padded_image_list, rewrite_list)
            return ret_list, dl, prep_t, net_t, post_t
        else:
            response = []
            bs = self.batch_size
            dl_t = 0
            prep_t = 0
            net_t = 0
            post_t = 0
            for i in range(0, len(idx), bs):
                prep_s = timer()
                padded_image_list, rewrite_list, dl = \
                    self.ds.load_samples_and_resize(idx[i:i+bs])
                prep_e = timer()
                prep_t += prep_e - prep_s - dl
                dl_t += dl
                ret_list, net, post = self.run_one_item(idx[i:i+bs],
                                                  padded_image_list,
                                                  rewrite_list)
                response.extend(ret_list)
                net_t += net
                post_t += post
            return response, dl_t, prep_t, net_t, post_t


def handle(event, _):
    from detect import RetinaFace

    batch_size = int(event['batch_size'])
    ml_start = timer()
    model = RetinaFace("/models/updated_withpreprop_r50.onnx",
                       "resnet50", batch_size,
                       1, "gpu")
    ml_end = timer()
    runner = Runner(model, DS, batch_size)
    responses, dl_t, prep_t, net_t, post_t = runner.run_samples(event['qs'])
    return {
        'ml': round(ml_end - ml_start, 2),
        'dl': round(dl_t, 2),
        'prep': round(prep_t, 2),
        'inf': round(net_t, 2),
        'post': round(post_t, 2),
    }


def main():
    length = len(DS.image_list)
    qs = [{'idx': x} for x in range(length)]
    event = {'qs': qs, 'batch_size': 16}
    ret = handle(event, None)
    print(ret)


if __name__ == '__main__':
    main()
