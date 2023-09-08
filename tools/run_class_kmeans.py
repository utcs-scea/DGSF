#!/usr/bin/env python3
import argparse
import sys
import os
import signal
import ctypes
import subprocess
from time import sleep

libc = ctypes.CDLL("libc.so.6")
def set_pdeathsig(sig = signal.SIGTERM):
    def callable():
        return libc.prctl(1, sig)
    return callable
 
def launch_ava_manager(ngpus, spec):
    cmd = f"""build/ava/release/svgpu_manager
        --worker_path build/ava/release/onnx_{spec}/bin/worker
        --manager_port 43201
        --worker_port_base 5100
        --worker_env LD_LIBRARY_PATH=/usr/local/cuda/nvvm/lib64
        --ngpus {ngpus}
        """
    return subprocess.Popen(cmd.split(), preexec_fn=set_pdeathsig(signal.SIGTERM))

def kill_ava(ava, spec):
    ava.terminate()
    ava.communicate()
    print("*** killed ava")
    sleep(1)
    cmd = f"pkill -f build/ava/release/onnx_{spec}/bin/worker"
    subprocess.call(cmd, shell=True)
    sleep(3)

def launch_process(cmd, working_dir, env_vars):
    p = subprocess.Popen(cmd.split(), env=env_vars, 
             cwd=working_dir, preexec_fn=set_pdeathsig(signal.SIGTERM))    
    _stdout, _stderr = p.communicate()
    if p.returncode != 0:
        print(f"!!! Process {p.pid} [cmd: {cmd}] returned non-zero code {p.returncode}")

def get_env(var_string):
    proc_env = os.environ.copy()
    envvars = var_string.split()
    for e in envvars:
        k,v = e.split("=")
        proc_env[k] = v
    return proc_env

def compile_kmeans(base_dir, kmeans):
    for kmeans_dir in kmeans:
        wd = base_dir + "/" + kmeans_dir
        launch_process("make", wd, get_env(""))

# current assumption: exec'ing kmeans in a directory two levels below main
def run_kmeans_serial(base_dir, kmeans, spec):
    cmd = f"./kmeans -k 16 -i inputs/small.txt -d 10 -s 12345 -g"

    for kmeans_dir in kmeans:
        wd = base_dir + "/" + kmeans_dir
        dump_dir = wd + "/" + "cuda_dumps"

        try:
            os.mkdir(dump_dir) # create dump directory
        except OSError as error:
            print(error)

        var_string = f"""
            LD_LIBRARY_PATH={os.getcwd()}/build/ava/release/onnx_dump/lib
            AVA_DUMP_DIR={dump_dir}
        """
        envvars = get_env(var_string)
        if "AVA_DUMP_DIR" in envvars.keys():
            print("launch with dump_dir:", envvars["AVA_DUMP_DIR"])
        launch_process(cmd, wd, envvars)
 
# current assumption: exec'ing in a directory two levels below main
def run_kmeans_concurrent(base_dir, kmeans):
    processes = []
    for kmeans_dir in kmeans:
        wd = base_dir + "/" + kmeans_dir
        cmd = f"./kmeans -k 16 -i inputs/small.txt -d 10 -s 12345 -g"
        var_string = f"""
            LD_LIBRARY_PATH={os.getcwd()}/build/ava/release/onnx_opt/lib
            AVA_DUMP_DIR={wd}/cuda_dumps
        """
        envvars = get_env(var_string)
        if "AVA_DUMP_DIR" in envvars.keys():
            print("launch with dump_dir:", envvars["AVA_DUMP_DIR"])

        p = subprocess.Popen(cmd.split(), env=envvars,
                cwd=wd, preexec_fn=set_pdeathsig(signal.SIGTERM))    
        processes.append(p)

    for p in processes:
        _stdout, _stderr = p.communicate()
        if p.returncode != 0:
            print(f"!!!!!!! Process {p.pid} returned non-zero code {p.returncode}")

def main():
    parser = argparse.ArgumentParser(description='Parse data from TCP log files')
    parser.add_argument('-r', '--rootdir', type=str, nargs=1,
                    help='path from which to get kmeans directories', required=True)
    parser.add_argument('-n', '--nruns', type=int, nargs='?', const=1,
                    help='number of times to run with opt spec', required=False)
    parser.add_argument('-c', '--compile', type=bool, nargs='?',
                    const=True, default=False, help="If set, compile all kmeans.")
    
    args = parser.parse_args()
    root_dir = args.rootdir[0]
    compile_all = args.compile

    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.join(script_dir, ".."))

    # get all student directories
    kmeans = [name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]

    if compile_all:
        compile_kmeans(rootdir, kmeans)

    # generate dumps
    #ava = launch_ava_manager(4, "dump")
    #run_kmeans_serial(root_dir, kmeans, "dump")
    #kill_ava(ava, "dump")

    # run all kmeans concurrently with opt spec
    for run in range (args.nruns):
        ava = launch_ava_manager(4, "opt")
        run_kmeans_concurrent(root_dir, kmeans)
        kill_ava(ava, "opt")


if __name__ == "__main__":
    main()
