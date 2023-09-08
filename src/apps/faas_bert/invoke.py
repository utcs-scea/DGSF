import argparse
import json
from timeit import default_timer as timer
import requests


class Runner:
    def __init__(self, url: str, batchsize: int, split: bool):
        self.url = url
        self.batchsize = batchsize
        self.split = split

    def issue_one_item(self, query_input):
        json_input = json.dumps(query_input)
        ret = requests.post(self.url, data=json_input, verify=False)
        if ret.status_code == requests.codes.ok:
            ret = ret.json()
            if ret:
                return ret['ml'], ret['dl'], ret['inf']

    def issue_queries(self, query_samples):
        queries = [{'idx': q['idx'], 'id': q['id']} for q in query_samples]
        if not self.split or (self.split and len(query_samples) < self.max_batchsize):
            input = {
                'qs': queries,
                'batch_size': self.batchsize,
            }
            return self.issue_one_item(input)
        else:
            ml_t = 0
            dl_t = 0
            inf_t = 0
            for i in range(0, len(query_samples), self.batchsize):
                sub_qs = queries[i:i+self.batchsize]
                ml, dl, inf = self.issue_one_item({
                    'qs': sub_qs,
                    'batch_size': self.batchsize,
                })
                ml_t += ml
                dl_t += dl
                inf_t += inf
            return ml_t, dl_t, inf_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--count', default=1000, type=int)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--split", action="store_true", help="whether to split the query samples into smaller pieces")
    args = parser.parse_args()
    full_url = args.url + '/function/faas_bert'
    runner = Runner(full_url, args.batchsize, args.split)
    qs = [{'idx': x, 'id': x+1} for x in range(args.count)]
    start = timer()
    ml_t, dl_t, inf_t = runner.issue_queries(qs)
    end = timer()
    print(f"Time: {round(end-start, 2)}")
    print(f"model loading: {ml_t}")
    print(f"data loading: {dl_t}")
    print(f"inf: {inf_t}")
    print(f"sample/s: {round(len(qs)/(end-start), 1)}")
    print(f"sample/s using inf: {round(len(qs)/inf_t, 1)}")


if __name__ == '__main__':
    main()
