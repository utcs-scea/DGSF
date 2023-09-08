import argparse
import json
import requests
from timeit import default_timer as timer


class Runner:
    def __init__(self, url: str, max_batchsize: int, split: bool):
        self.url = url
        self.max_batchsize = max_batchsize
        self.split = split

    def issue_one_item(self, query_input):
        json_input = json.dumps(query_input)
        ret = requests.post(self.url, data=json_input, verify=False)
        if ret.status_code == requests.codes.ok:
            ret = ret.json()
            if ret:
                return ret['ml'], ret['dl'], ret['inf']
        return None, None, None

    def issue_queries(self, query_samples):
        queries = [{'idx': q['idx'], 'id': q['id']} for q in query_samples]
        if not self.split or (len(query_samples) < self.max_batchsize and self.split):
            input = {
                'qs': queries,
                'batch_size': self.max_batchsize,
            }
            ml, dl, inf = self.issue_one_item(input)
            return ml, dl, inf
        else:
            ml_t = 0
            dl_t = 0
            inf_t = 0
            for i in range(0, len(query_samples), self.max_batchsize):
                sub_qs = queries[i:i+self.max_batchsize]
                ml, dl, inf = self.issue_one_item({
                    'qs': sub_qs,
                    'batch_size': self.max_batchsize,
                })
                ml_t += float(ml)
                dl_t += float(dl)
                inf_t += float(inf)
            return ml_t, dl_t, inf_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=16)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--split", action="store_true",
                        help="whether to split the query samples into smaller pieces")
    parser.add_argument("--count", default=24576, type=int)
    args = parser.parse_args()
    full_url = args.url + '/function/faas_classification_detection'
    start = timer()
    runner = Runner(full_url, args.batchsize, args.split)
    qs = [{'idx': x, 'id': x+1} for x in range(args.count)]
    ml, dl, inf = runner.issue_queries(qs)
    end = timer()
    print(f"Time: {end-start} s")
    print(f"model loading: {ml}")
    print(f"data loading: {dl}")
    print(f"inf: {inf}")
    print(f"sample/s: {len(qs)/(end-start)}")
    print(f"sample/s using inf: {len(qs)/inf}")


if __name__ == '__main__':
    main()
