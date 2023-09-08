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
from random import shuffle


ENDPOINT_PREFIX = "http://127.0.0.1:8080/function/"
deploy_path = "../../serverless/tools/functions/gen_yaml_and_deploy.py"
image_path = "../../build/serverless/images"
apps_path = "../../src/apps/"

gpu_mem = 1024
workloads = ["uniform_kmeans"]
workload_funcs = {"uniform_kmeans": 10}
N = 1

def post_request(req_data, function):
    r = requests.post(url = ENDPOINT_PREFIX + function, data = json.dumps(req_data))

# if we want to do any cleaning up
def signal_handler(sig, frame):
    print('pressed Ctrl+C')
    sys.exit(0)

def uniform_kmeans_run():
    print("in kmeans_run")
    # get all student directories
    kmeans_path = os.path.join(apps_path, "uniform_kmeans")
    kmeans = [name for name in os.listdir(kmeans_path) \
            if (os.path.isdir(os.path.join(kmeans_path, name)) and "kmeans1" in name)]

    threads = []
    for kmeans_dir in kmeans:
        data = {'k':kmeans_dir,
                'i':'inputs/1000000p-10d.txt',
                'n':1,
                'd':10}

        t = threading.Thread(target=post_request, args=(data,"uniform_kmeans"), daemon=True)
        t.start()
        threads.append(t)

    return threads

def main():
    parser = argparse.ArgumentParser(description='Run N serverless functions')
    parser.add_argument('-l', '--lamda', type=int, default=0,
                    help='lamda to deploy functions with poisson distribution', required=False)
    
    args = parser.parse_args()
    signal.signal(signal.SIGINT, signal_handler)

    n_functions = len(workloads) * N

    # for launching kmeans at a poisson-distributed rate (sec)
    np.random.seed(865809)
    sleep_times = random.poisson(args.lamda, n_functions)

    # create random order of functions * N
    functions_list = list(np.repeat(workloads, N))
    random.shuffle(functions_list)

    # pre-deploy functions
    for wkld in workloads:
        n_funcs = workload_funcs[wkld]
        cmd = [deploy_path, image_path, wkld, "--gpumem", str(gpu_mem), "--n", str(n_funcs)]
        rc = subprocess.call(cmd)
        sleep(n_funcs * 1.5)

    threads = []
    sleep_cycle = cycle(sleep_times)
    start = timer()

    # launch functions
    for function in functions_list:
        thread_list = eval(function + "_run()")
        threads.extend(thread_list)
        wait_time = next(sleep_cycle)
        print("wait time:", wait_time)
        sleep(wait_time)

    for t in threads:
        t.join()

    end = timer()   
    print("total:", end-start)
            
if __name__ == "__main__":
    main()
