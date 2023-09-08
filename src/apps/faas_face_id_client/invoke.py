import argparse
import json
import logging
from timeit import default_timer as timer
import requests
from dataset import LFWDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default="http://127.0.0.1:8080", help="gateway url")
    parser.add_argument('--pairs-list', required=True)
    parser.add_argument('--dataset-path', default='./input_data/lfw_faces/')
    parser.add_argument('--batchsize', default=16)
    parser.add_argument('--count', type=int)
    args = parser.parse_args()
    ds = LFWDataset(args.dataset_path, args.pairs_list)
    queries = [{'idx': x} for x in range(len(ds.nameLs))]
    if args.count is not None:
        queries = queries[:args.count]
    event = {'qs': queries, 'batch_size': args.batchsize}
    json_input = json.dumps(event)
    full_url = args.url + '/function/faas_face_id'
    start = timer()
    ret = requests.post(full_url, data=json_input, verify=False)
    end = timer()
    if ret.status_code == requests.codes.ok:
        ret = ret.json()
        if ret:
            print(f'end to end: {round(end-start, 2)}')
            print(f"data loading: {ret['dl']}")
            print(f"model loading: {ret['ml']}")
            print(f"inf: {ret['inf']}")
            id_list = [x for x in range(len(ds.nameLs))]
            _, _, flags = ds.get_samples(id_list)
            correct = 0
            for i, sim in enumerate(ret['res']):
                if sim > 0.5:
                    if flags[i]:
                        correct += 1
                    else:
                        logging.debug("{} and {} should not match".format(
                            ds.nameLs[id_list[i]],
                            ds.nameRs[id_list[i]]))
                else:
                    if flags[i]:
                        logging.debug("{} and {} should match".format(
                            ds.nameLs[id_list[i]],
                            ds.nameRs[id_list[i]]))
                    else:
                        correct += 1
            print(f"{correct} / {len(queries)} correct")
    print(f'sample/s: {round(len(queries)/(end-start), 1)}')


if __name__ == '__main__':
    main()
