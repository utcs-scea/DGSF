#!/usr/bin/env python3
import subprocess
import sys, os
from time import sleep
from timeit import default_timer as timer
import argparse
import signal
 
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
 
def run_kmeans(kmeans, dims, infile, ngpus):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    processes = []
    procmap = {}
    device = 0
    for kmeans_dir in kmeans:
        infile_path = os.path.join(this_dir, infile)
        wd = os.path.join(this_dir, kmeans_dir)
        kmeans_dir = os.path.join(this_dir, kmeans_dir)

        with open(os.path.join(kmeans_dir, "executable")) as f:
            exec_cmd = f.read().splitlines()[0]

        cmd = f"{exec_cmd} -k 16 -i {this_dir}/{infile} -d {dims} -t 0.01 -m 200 -s 8675309"
        cmd = cmd.replace("\n", " ")
        print("> cmd:", cmd)
        print("> kmeans:", kmeans_dir)

        visible_devices = f"CUDA_VISIBLE_DEVICES={device}"
        device =(device + 1) % ngpus
        print("launch with device", device)

        envvars = get_env(visible_devices)
        p = subprocess.Popen(cmd.split(), env=envvars, cwd=wd)
        processes.append(p)
        procmap[p] = kmeans_dir
        sleep(1)

    for p in processes:
        _stdout, _stderr = p.communicate()
        if p.returncode != 0:
            print(f"!!!!!!! Process {p.pid}, dir={procmap[p]} returned non-zero code {p.returncode}")
        else:
            print("done:", procmap[p])
         
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
    parser.add_argument('-n', '--nruns', type=int, nargs='?', const=1,
                    help='number of times to run', required=False)
    
    args = parser.parse_args()
    infile = args.input[0]
    dims = args.dims[0]
    n_runs = args.nruns

    sys.stdout = open('stdout', 'w')
    signal.signal(signal.SIGINT, signal_handler)

    # get all student directories
    kmeans = [name for name in os.listdir(".") \
            if (os.path.isdir(os.path.join(".", name)) and 
                "kmeans" in name)]
#                ("kmeans-16" in name
#                or "kmeans-17" in name or "kmeans-18" in name or "kmeans-19" in name
#                or "kmeans-20" in name or "kmeans-21" in name or "kmeans-22" in name
#                or "kmeans-23" in name or "kmeans-24" in name or "kmeans-26" in name
#                or "kmeans-28" in name or "kmeans-29" in name or "kmeans-30" in name
#                or "kmeans-32" in name or "kmeans-33" in name or "kmeans-34" in name
#                or "kmeans-36" in name or "kmeans-37" in name or "kmeans-40" in name
#                or "kmeans-41" in name or "kmeans-42" in name or "kmeans-43" in name
               # or "kmeans-45" in name or "kmeans-48" in name or "kmeans-49" in name
#                or "kmeans-59" in name or "kmeans-6" in name or "kmeans-61" in name
               # or "kmeans-62" in name or "kmeans-63" in name or "kmeans-64" in name
#            ))]
 
    for run in range(n_runs): 
        f = open(f"baseline-{run}.txt", "w")
        print("run", run)

        start = timer()
        
        run_kmeans(kmeans, dims, infile, args.ngpus)

        end = timer()
        f.write(str(end-start))
        f.write("\n")
      
        sleep(1)

    f.close()


if __name__ == "__main__":
    main()