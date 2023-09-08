#!/usr/bin/env python3
import subprocess
import sys, os
from time import sleep
from timeit import default_timer as timer
import argparse
import signal
import numpy as np
import threading

all_kmeans = []

kmeans_assign = []

def run_kmeans(kmeans, dims, infile, gpu):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    for kmeans_dir in kmeans:
        infile_path = os.path.join(this_dir, infile)
        wd = os.path.join(this_dir, kmeans_dir)
        kmeans_dir = os.path.join(this_dir, kmeans_dir)

        with open(os.path.join(kmeans_dir, "executable")) as f:
            exec_cmd = f.read().splitlines()[0]

        ddir = os.path.join(this_dir, kmeans_dir, "cuda_dumps")
        env = f"AVA_GUEST_DUMP_DIR={ddir} AVA_WORKER_DUMP_DIR={ddir} "

        env += f"AVA_CONFIG_FILE_PATH={os.path.join(this_dir, '../../../tools/ava.conf')} "
        env += f"LD_LIBRARY_PATH={os.path.join(this_dir, '../../../build/ava/release/onnx_opt/lib')} "
        
        cmd = f"{env} {os.path.join(wd, exec_cmd.lstrip())} -k 16 -i {this_dir}/{infile} -d {dims} -t 0 -m 100 -s 8675309 "
        print("> cmd:", cmd)
        print("> kmeans:", kmeans_dir)

        input("Press Enter to continue...")
        p = subprocess.run(cmd, cwd=wd, shell=True)

        #sleep(1)

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.stdout.close()
    sys.exit(0)

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

def main():
    parser = argparse.ArgumentParser(description='run all class kmeans on host')
    parser.add_argument('-i', '--input', type=str,
                    help='path from which to get input data', required=True)
    parser.add_argument('-d', '--dims', type=int, 
                    help='number of dimensions of input points', required=True)
    parser.add_argument('-n', '--nruns', type=int, nargs='?', const=1,
                    help='number of times to run', required=False)
    
    args = parser.parse_args()
    infile = args.input
    dims = args.dims
    n_runs = args.nruns

    #sys.stdout = open('baseline-stdout', 'w')
    signal.signal(signal.SIGINT, signal_handler)

    # get all student directories
    kmeans = [name for name in os.listdir(".") \
            if (os.path.isdir(os.path.join(".", name)) and 
                "kmeans17" in name)]
    print(kmeans)

    # first_split = split_list(kmeans)
    # first_half_split = split_list(first_split[0])
    # second_half_split = split_list(first_split[1])
    # kmeans_split = first_half_split + second_half_split

    f = open(f"serial-baseline.txt", "w")
    for run in range(n_runs): 
        print("run", run)

        start = timer()
        # threads = []
        # for gpu in range(1):
        #     print("launching gpu", gpu,"with kmeans", kmeans_split[gpu])
        #     t = threading.Thread(target=run_kmeans, args=(kmeans_split[gpu], dims, infile, gpu), daemon=True)
        #     t.start()
        #     threads.append(t)

        # for t in threads:
        #     t.join()

        run_kmeans(kmeans, dims, infile, 0)

        end = timer()
        
        print("time: ", end-start)
        
        f.write(str(end-start))
        f.write("\n")

    f.close()

    print("all:", all_kmeans)


if __name__ == "__main__":
    main()
