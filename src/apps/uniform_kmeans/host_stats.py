#!/usr/bin/env python3
import subprocess
import sys, os
from time import sleep
from timeit import default_timer as timer
import argparse
import signal
from statistics import mean
from statistics import stdev
 
def get_env(var_string):
    proc_env = os.environ.copy()
    envvars = var_string.split()
    for e in envvars:
        k,v = e.split("=")
        proc_env[k] = v
    return proc_env

def signal_handler(sig, frame):
    sys.stdout.close()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='run all class kmeans on host')
    parser.add_argument('-i', '--input', type=str,
                    help='path from which to get input data', required=True)
    parser.add_argument('-d', '--dims', type=int,
                    help='number of dimensions of input points', required=True)
    
    args = parser.parse_args()
    infile = args.input
    dims = args.dims

    #sys.stdout = open('stats-stdout', 'w')
    signal.signal(signal.SIGINT, signal_handler)

    # get all student directories
    kmeans = [name for name in os.listdir(".") \
            if (os.path.isdir(os.path.join(".", name)) and 
                "kmeans-" in name)]

    this_dir = os.path.dirname(os.path.realpath(__file__))

    device = 1
    times = []
    kmap = {}

    for kmeans_dir in kmeans:
        infile_path = os.path.join(this_dir, infile)
        wd = os.path.join(this_dir, kmeans_dir)
        kmeans_dir = os.path.join(this_dir, kmeans_dir)

        with open(os.path.join(kmeans_dir, "executable")) as f:
            exec_cmd = f.read().splitlines()[0]

        cmd = f"{exec_cmd} -k 16 -i {this_dir}/{infile} -d {dims} -t 0.01 -m 200 -s 8675309"
        cmd = cmd.replace("\n", " ")
        #print("> cmd:", cmd)
        #print("> kmeans:", kmeans_dir)

        visible_devices = f"CUDA_VISIBLE_DEVICES={device}"
        envvars = get_env(visible_devices)

        start = timer()
        p = subprocess.Popen(cmd.split(), env=envvars, cwd=wd)
        _stdout, _stderr = p.communicate()
        end = timer()
        if p.returncode != 0:
            print(f"!!!!!!! Process {p.pid}, dir={kmeans_dir} returned non-zero code {p.returncode}")
        else:
            print(kmeans_dir, ":", str(end-start))
        times.append(end-start)
        kmap[kmeans_dir] = end-start

    print("map:", kmap) # used for debugging
    #print("all times:", times)
    print("min:", min(times))
    print("max:", max(times))
    print("mean:", mean(times))
    print("stdev:", stdev(times))
 
if __name__ == "__main__":
    main()
