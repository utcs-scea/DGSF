#!/usr/bin/env python3
import argparse
import signal
import os
import subprocess
from timeit import default_timer as timer
from time import sleep
import requests
import json
import threading
import sys
import numpy as np
from numpy import random
from itertools import cycle


ENDPOINT = "http://127.0.0.1:8080/function/class_kmeans"
lam = 27.4936 / 4
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

def signal_handler(sig, frame):
    print('pressed Ctrl+C')
    sys.stdout.close()
    sys.exit(0)

def main():

    parser = argparse.ArgumentParser(description='Parse data from TCP log files')
    parser.add_argument('-i', '--input', type=str,
                    help='path from which to get input data', required=True)
    parser.add_argument('-k', '--kmeans', type=str,
                    help='kmeans dir', required=True)
    parser.add_argument('-d', '--dims', type=int, 
                    help='number of dimensions of input points', required=True)
    parser.add_argument('-f', '--functions', type=int, 
                    help='number of functions to pre-deploy', required=False)
    parser.add_argument('-p', '--poisson', type=bool, nargs='?',
                    const=True, default=False, help="If set, launch kmeans at poisson-dist intervals.")
    
    args = parser.parse_args()
    infile = args.input
    dims = args.dims
    kmeans_dir = args.kmeans

    # get all student directories
    kmeans = [name for name in os.listdir(".") \
            if (os.path.isdir(os.path.join(".", name)) and kmeans_dir in name)]

    functions = len(kmeans) + 5
    if args.functions is not None:
        functions = args.functions

    # for launching kmeans at a poisson-distributed rate (sec)
    np.random.seed(865809)
    run_poisson = False
    if args.poisson:
        run_poisson = True
    sleep_times = random.poisson(lam, len(kmeans))
    print("lambda:", lam)
    print("gpumem:", gpu_mem)

    # pre-deploy functions
    cmd = [deploy_path, image_path, "class_kmeans", "--gpumem", str(gpu_mem), "--n", str(functions)]
    rc = subprocess.call(cmd)

    #sys.stdout = open('faas-times.txt', 'w')
    signal.signal(signal.SIGINT, signal_handler)

    sleep(functions * 1.5)

    threads = []
    kmeansdict = {}
    sleep_cycle = cycle(sleep_times)
    start = timer()

    # launch kmeans
    for kmeans_dir in kmeans:
        print("launching", kmeans_dir)
        data = {'k':kmeans_dir,
                'i':infile,
                'n':1,
                'd':dims}

        t = threading.Thread(target=post_request, args=(data,), daemon=True)
        t.start()
        threads.append(t)
        kmeansdict[t] = kmeans_dir

        if args.poisson:
            wait_time = next(sleep_cycle)
            print("wait time:", wait_time)
            sleep(wait_time)
        else:
            sleep(.02)

    for t in threads:
        print("joining:", kmeansdict[t])
        try:
            t.join()
        except Exception as e:
            print("Exception in thread", kmeansdict[t], ":", e)

    end = timer()   
    print(totals)
    print("total:", end-start)
            
if __name__ == "__main__":
    main()
