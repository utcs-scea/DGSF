import os
os.environ['FRAMEWORK'] = 'onnxrt'
from timeit import default_timer as timer
import numpy as np
from arcface import ArcFace
from dataset import LFWDataset

DS = LFWDataset('/data/', '/data/pairs.txt')


def handle(event, _):
    id_list = [q['idx'] for q in event['qs']]
    start = timer()
    nameL_imgs, nameR_imgs, flags = DS.get_samples(id_list)
    ml_start = timer()
    model = ArcFace('/models/updated_arcfaceresnet100-8.onnx',
                    int(event['batch_size']), 1, "gpu")
    ml_end = timer()
    left_embeddings = model.inference_batch(nameL_imgs)
    right_embeddings = model.inference_batch(nameR_imgs)
    sim_arr = np.sum(np.multiply(left_embeddings, right_embeddings), 1)
    end = timer()
    return {
        'res': sim_arr.tolist(),
        'dl': round(ml_start - start, 2),
        'ml': round(ml_end - ml_start, 2),
        'inf': round(end-ml_end, 2),
    }


def main():
    qs = [{'idx': i} for i in range(0, len(DS.nameLs))]
    event = {
        'qs': qs,
        'batch_size': 16
    }
    ret = handle(event, None)
    id_list = [q['idx'] for q in event['qs']]
    _, _, flags = DS.get_samples(id_list)
    correct = 0

    for i, sim in enumerate(ret['res']):
        if sim > 0.5:
            if flags[i]:
                correct += 1
            else:
                print("{} and {} should not match".format(DS.nameLs[id_list[i]], DS.nameRs[id_list[i]]))
        else:
            if flags[i]:
                print("{} and {} should match".format(DS.nameLs[id_list[i]], DS.nameRs[id_list[i]]))
            else:
                correct += 1
    print(f"{correct} / {len(id_list)} correct")


if __name__ == '__main__':
    main()
