import argparse
import json
from timeit import default_timer as timer
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080",
                        help="gateway url")
    parser.add_argument('--imglist', required=True)
    parser.add_argument('--batchsize', default=16)
    parser.add_argument('--count', type=int, help="limit number of images", default=0)
    args = parser.parse_args()

    count = args.count
    image_list = []
    with open(args.imglist, 'r') as f:
        for s in f:
            image_name = s.strip()
            image_list.append(image_name)
    queries = [{'idx': x} for x in range(len(image_list))]

    if count != 0:
        queries = queries[:count]

    event = {'qs': queries, 'batch_size': args.batchsize}
    
    json_input = json.dumps(event)
    full_url = args.url + '/function/faas_face_det'
    start = timer()
    ret = requests.post(full_url, data=json_input, verify=False)
    end = timer()
    if ret.status_code == requests.codes.ok:
        ret = ret.json()
        if ret:
            # print(ret)
            print(f"model loading: {ret['ml']} s")
            print(f"data loading: {ret['dl']} s")
            print(f"pre proc: {ret['prep']}")
            print(f"inf: {ret['inf']} s")
            print(f"post proc: {ret['post']}")
    print(f"end to end: {round(end-start, 2)} s")
    print(f'samples/s: {round(len(queries)/(end - start), 1)}')


if __name__ == '__main__':
    main()
