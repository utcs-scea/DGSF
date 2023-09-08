#!/usr/bin/env python3
import subprocess
import sys, os
from time import sleep
from timeit import default_timer as timer
import argparse
import signal
from itertools import cycle
import numpy as np
from numpy import random

#lam = 17.19 / 4
#lam = 27.4936 / 4
lam = 2
 
def get_env(var_string):
    proc_env = os.environ.copy()
    envvars = var_string.split()
    for e in envvars:
        k,v = e.split("=")
        proc_env[k] = v
    return proc_env

def get_cuda_cmd(wd, cmd_id):
    with open(os.path.join(wd, "submit")) as f:
        lines = f.readlines()
        for i in range(0, len(lines)):
            line = lines[i]
            if "[CUDA basic]" in line:
                while cmd_id not in lines[i]:
                    i += 1
                extra_args = ""
                if "Executable" in cmd_id:
                    extra_args += lines[i+1].split(":")[1]
                return lines[i].split(":")[1] + extra_args
 
def run_kmeans(kmeans, dims, infile, ngpus, poisson):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    processes = []
    procmap = {}
    device = 0

    # for launching kmeans at a poisson-distributed rate (sec)
    np.random.seed(865809)
    run_poisson = False
    if poisson:
        run_poisson = True
    sleep_times = random.poisson(lam, len(kmeans))
    print("lambda:", lam)

    sleep_cycle = cycle(sleep_times)
   
    for kmeans_dir in kmeans:
        infile_path = os.path.join(this_dir, infile)
        wd = os.path.join(this_dir, kmeans_dir)
        kmeans_dir = os.path.join(this_dir, kmeans_dir)

        #with open(os.path.join(kmeans_dir, "executable")) as f:
        #    exec_cmd = f.read().splitlines()[0]

        #cmd = f"{exec_cmd} -k 16 -i {this_dir}/{infile} -d {dims} -t 0.01 -m 200 -s 8675309"
        #cmd = cmd.replace("\n", " ")
        cmd = f"./handler.py -k {kmeans_dir} -i {infile} -d {dims} -n 1"
        print("> cmd:", cmd)
        print("> kmeans:", kmeans_dir)

        visible_devices = f"CUDA_VISIBLE_DEVICES={device}"
        device = (device + 1) % ngpus
        print("launch with device", device)

        envvars = get_env(visible_devices)
        #p = subprocess.Popen(cmd.split(), env=envvars, cwd=wd)
        p = subprocess.Popen(cmd.split(), env=envvars)
        processes.append(p)
        procmap[p] = kmeans_dir
        if (poisson):
            sleep_t = next(sleep_cycle)
            print("sleep for", sleep_t)
            sleep(sleep_t)
        else:
            sleep(.02)

    for p in processes:
        _stdout, _stderr = p.communicate()
        if p.returncode != 0:
            print(f"!!!!!!! Process {p.pid}, dir={procmap[p]} returned non-zero code {p.returncode}")
        #else:
        #    print("done:", procmap[p])
         
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.stdout.close()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='run all class kmeans on host')
    parser.add_argument('-i', '--input', type=str, nargs=1,
                    help='path from which to get input data', required=True)
    parser.add_argument('-d', '--dims', type=int, nargs=1,
                    help='number of dimensions of input points', required=True)
    parser.add_argument('-g', '--ngpus', type=int,
                    help='number of gpus to use', required=False, default=1)
    parser.add_argument('-n', '--nruns', type=int, nargs='?', default=1,
                    help='number of times to run', required=False)
    parser.add_argument('-p', '--poisson', type=bool, nargs='?',
                    const=True, default=False, help="If set, launch kmeans at poisson-dist intervals.")
    
    args = parser.parse_args()
    infile = args.input[0]
    dims = args.dims[0]
    n_runs = args.nruns

    #sys.stdout = open('stdout', 'w')
    #signal.signal(signal.SIGINT, signal_handler)

    # get all student directories
    kmeans = [name for name in os.listdir(".") \
            if (os.path.isdir(os.path.join(".", name)) and 
                "kmeans" in name)]
 
    f = open(f"concurrent-baseline.txt", "w")
    for run in range(n_runs): 
        print("run", run)

        start = timer()
        
        if args.poisson:
            print("run with poisson")
        run_kmeans(kmeans, dims, infile, args.ngpus, args.poisson)

        end = timer()
        f.write(str(end-start))
        f.write("\n")
      
        sleep(1)

    f.close()


if __name__ == "__main__":
    main()
