#!/usr/bin/env python3
import argparse
import sys
import os
import subprocess
from time import sleep
from timeit import default_timer as timer

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

def setup_run(kmeans_dir, infile, dims, spec):            
    #host_dir = "/local_disk/eyoon/serverless-gpus/src/apps/uniform_kmeans"
    host_dir = "/disk/hfingler/serverless-gpus/src/apps/uniform_kmeans"
    this_dir = os.path.dirname(os.path.realpath(__file__))

    guest_dump_dir = os.path.join(this_dir, kmeans_dir, "cuda_dumps")
    worker_dump_dir = os.path.join(host_dir, kmeans_dir, "cuda_dumps")
    infile_base = os.path.basename(infile)

    with open(os.path.join(this_dir, kmeans_dir, "executable")) as f:
        exec_cmd = f.read().splitlines()[0]

    exec_cmd = os.path.join(this_dir, kmeans_dir, exec_cmd)
    exec_cmd = exec_cmd.replace(" ", "")

    cmd = f"{exec_cmd} -k 16 -i {this_dir}/{infile} -d {dims} -t 0.001 -m 200 -s 8675309 "
    print("> cmd:", cmd)

    var_string = f"AVA_GUEST_DUMP_DIR={guest_dump_dir} AVA_WORKER_DUMP_DIR={worker_dump_dir}"
    envvars = get_env(var_string)
    
    print("> Launch worker with dump_dir:", envvars["AVA_WORKER_DUMP_DIR"])
    print("> Launch guest with dump_dir:", envvars["AVA_GUEST_DUMP_DIR"])
    return (cmd, os.path.join(this_dir, kmeans_dir), envvars)


def run_kmeans(kmeans_dir, infile, dims, nruns):
    runtimes = []
    for run in range(nruns):
        processes = []
        (cmd, wd, envvars) = setup_run(kmeans_dir, infile, dims, "opt")

        start = timer()
        p = subprocess.Popen(cmd.split(), env=envvars, cwd=wd, stdout=subprocess.DEVNULL)
        _stdout, _stderr = p.communicate()
        end = timer()

        if p.returncode != 0:
            print(f"> !!!!!!! Process {p.pid} returned non-zero code {p.returncode}")

        runtimes.append(end-start)

    return runtimes

def main():
    parser = argparse.ArgumentParser(description='Parse data from TCP log files')
    parser.add_argument('-k', '--kmeansdir', type=str, nargs=1,
                    help='kmeans directory to run', required=True)
    parser.add_argument('-i', '--input', type=str, nargs=1,
                    help='path from which to get input data', required=True)
    parser.add_argument('-d', '--dims', type=int, nargs=1,
                    help='number of dimensions of input points', required=True)
    parser.add_argument('-n', '--nruns', type=int, nargs='?', const=1,
                    help='number of times to run kmeans', required=False)

    args = parser.parse_args()
    kmeans_dir = args.kmeansdir[0]
    infile = args.input[0]
    dims = args.dims[0]

    runtimes = run_kmeans(kmeans_dir, infile, dims, args.nruns)
    return runtimes

def handle(event, _):
    k = event["k"]
    i = event["i"]
    d = event["d"]
    n = event["n"]

    runtimes = run_kmeans(k, i, int(d), int(n))
    return {"result": runtimes}

if __name__ == "__main__":
    main()
