#!/usr/bin/env python3
import argparse
import os
import subprocess
from timeit import default_timer as timer
from time import sleep
import requests
import json
import threading
from statistics import mean
from statistics import stdev
import pprint

ENDPOINT = "http://127.0.0.1:8080/function/uniform_kmeans"
deploy_path = "./../../../serverless/tools/functions/gen_yaml_and_deploy.py"
image_path = "../../../build/serverless/images"
gpu_mem = 1024

totals = {}

def post_request(req_data):
    r = requests.post(url = ENDPOINT, data = json.dumps(req_data))
    str_result = r.content.decode("utf-8")
    dict_result = json.loads(str_result)
    print(req_data["k"], ":", dict_result["result"])
    totals[req_data["k"]] = dict_result["result"]

def main():

    parser = argparse.ArgumentParser(description='Parse data from TCP log files')
    parser.add_argument('-i', '--input', type=str,
                    help='path from which to get input data', required=True)
    parser.add_argument('-d', '--dims', type=int, 
                    help='number of dimensions of input points', required=True)
    
    args = parser.parse_args()
    infile = args.input
    dims = args.dims

    # get all student directories
    kmeans = [name for name in os.listdir(".") \
            if (os.path.isdir(os.path.join(".", name)) and "kmeans-54" in name)]

    cmd = [deploy_path, image_path, "uniform_kmeans", "--gpumem", str(gpu_mem), "--n", str(1)]
    rc = subprocess.call(cmd)

    sleep(4)
    times = []
    time_dict = {}

    pprint.pprint(time_dict)
    for kmeans_dir in kmeans:
        print("launching", kmeans_dir)
        data = {'k':kmeans_dir,
                'i':infile,
                'n':1,
                'd':dims}

        start = timer()
        t = threading.Thread(target=post_request, args=(data,), daemon=True)
        t.start()
        t.join()
        end = timer()
        print("done:", kmeans_dir)
        times.append(end-start)
        time_dict[kmeans_dir] = end-start

    pprint.pprint(time_dict)

    dict_times = totals.values()
    print()
    print("all times:", totals)
    print()
    print("---------------------------------------")
    print()
    print("min f:", min(dict_times))
    print("max f:", max(dict_times))
    print("mean f:", mean(dict_times))
    print("stdev f:", stdev(dict_times))
    print()
    print("min:", min(times))
    print("max:", max(times))
    print("mean:", mean(times))
    print("stdev:", stdev(times))

            
if __name__ == "__main__":
    main()
