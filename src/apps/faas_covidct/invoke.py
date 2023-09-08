import argparse
import json
from timeit import default_timer as timer
import requests

from step1 import get_patient_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="gateway url")
    parser.add_argument('--batchsize', default=16)
    parser.add_argument('--plist', required=True, help='path to patient list')
    parser.add_argument('--count', type=int, help="limit number of images", default=0)
    args = parser.parse_args()
    count = args.count

    full_url = args.url + '/function/faas_covidct'
    
    
    plist = get_patient_list(args.plist)
    qs = [{'idx': x} for x in range(len(plist))]
    if count != 0:
        qs = qs[:count]

    event = {'qs': qs, 'batch_size': args.batchsize}
    json_input = json.dumps(event)
    start = timer()
    ret = requests.post(full_url, data=json_input, verify=False)
    end = timer()
    if ret.status_code == requests.codes.ok:
        print(f'end to end: {end - start}')
        ret = ret.json()
        if ret:
            print(f"bcdu model loading: {ret['bcdu_ml']}")
            print(f"data loading: {ret['dl']}")
            print(f"zoom kernel: {ret['zi']}")
            print(f"bcdu inf: {ret['bcdu_inf']}")
            print(f"other prep: {ret['other_prep']}")
            print(f"cnn model loading: {ret['cnn_ml']}")
            print(f"cnn inf: {ret['cnn_inf']}")
    print(f'sample/s: {round(len(qs)/(end - start), 1)}')


if __name__ == '__main__':
    main()
