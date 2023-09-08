#!/usr/bin/env python3
import argparse
import sys
import os
import signal
import ctypes
import subprocess
from time import sleep

successful = []

libc = ctypes.CDLL("libc.so.6")
def set_pdeathsig(sig = signal.SIGTERM):
    def callable():
        return libc.prctl(1, sig)
    return callable
 
def launch_ava_manager(ngpus, spec):
    cmd = f"""build/ava/release/svgpu_manager
        --worker_path build/ava/release/onnx_{spec}/bin/worker
        --manager_port 43200                 2
        --worker_port_base 5300
        --worker_env LD_LIBRARY_PATH=/usr/local/cuda/nvvm/lib64
        --ngpus {ngpus}
        """
    return subprocess.Popen(cmd.split(), preexec_fn=set_pdeathsig(signal.SIGTERM))

def kill_ava(ava, spec):
    ava.terminate()
    ava.communicate()
    print("> Killed ava")
    sleep(1)
    cmd = f"pkill -f build/ava/release/onnx_{spec}/bin/worker"
    subprocess.call(cmd, shell=True)
    sleep(3)

def launch_process(cmd, working_dir, env_vars):
    p = subprocess.Popen(cmd.split(), env=env_vars, 
             cwd=working_dir, preexec_fn=set_pdeathsig(signal.SIGTERM))    
    _stdout, _stderr = p.communicate()
    if p.returncode != 0:
        print(f"> !!! Process {p.pid} in wd {working_dir} [cmd: {cmd}] returned non-zero code {p.returncode}")
        exit(-1)

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

def compile_kmeans(base_dir, kmeans):
    for kmeans_dir in kmeans:
        wd = base_dir + "/" + kmeans_dir
        compile_cmd = get_cuda_cmd(wd, "How_To_Compile")
        print("> Compiling # ", kmeans_dir)
        print("> Compile command:", compile_cmd)
        launch_process(compile_cmd, wd, get_env(""))

def setup_run(base_dir, kmeans_dir, infile, dims, spec):            
    wd = os.path.join(os.getcwd(), base_dir, kmeans_dir)
    dump_dir = os.path.join(wd, "cuda_dumps")
    input_dir = os.path.join(wd, "inputs")
    infile_base = os.path.basename(infile)

    with open(os.path.join(wd, "executable")) as f:
        exec_cmd = f.read().splitlines()[0]

    cmd = f"{exec_cmd} -k 16 -i {os.path.join(wd, infile)} -d {dims} -t 0.01 -m 200 -s 8675309"
    cmd = cmd.replace("\n", " ")
    print("> cmd:", cmd)
    print("> kmeans:", kmeans_dir)

    var_string = f"""
        LD_LIBRARY_PATH={os.getcwd()}/build/ava/release/onnx_{spec}/lib
        AVA_CONFIG_FILE_PATH={os.getcwd()}/tools/ava.conf
        AVA_GUEST_DUMP_DIR={dump_dir}
        AVA_WORKER_DUMP_DIR={dump_dir}
    """
    envvars = get_env(var_string)
    print("> Launch with guest dump_dir:", envvars["AVA_GUEST_DUMP_DIR"])
    print("> Launch with worker dump_dir:", envvars["AVA_WORKER_DUMP_DIR"])
    return (cmd, wd, envvars)

def run_kmeans_serial(base_dir, kmeans, infile, dims):
    for kmeans_dir in kmeans:
        print("launch", kmeans_dir)
        (cmd, wd, envvars, dump_dir) = setup_run(base_dir, kmeans_dir, infile, dims, "dump")
        launch_process(cmd, wd, envvars)
        os.system(f"sudo mv /tmp/*.ava {dump_dir}")

def run_kmeans_concurrent(base_dir, kmeans, infile, dims):
    global successful
    processes = []
    nameMap = {}
    for kmeans_dir in kmeans:
        (cmd, wd, envvars) = setup_run(base_dir, kmeans_dir, infile, dims, "opt")

        p = subprocess.Popen(cmd.split(), env=envvars,
                cwd=wd, preexec_fn=set_pdeathsig(signal.SIGTERM))    
        nameMap[p] = kmeans_dir
        processes.append(p)
        sleep(2)

    for p in processes:
        _stdout, _stderr = p.communicate()
        if p.returncode != 0:
            print(f"> !!!!!!! Process {p.pid} returned non-zero code {p.returncode}")
            print("failed for directory", nameMap[p])
        else:
            successful.append(nameMap[p])

def signal_handler(sig, frame):
    global successful
    print("Successful processes:")
    for s in sorted(successful):
        print(s)
    sys.stdout.close()
    sys.exit(0)


# current assumption: this script is ran from the top directory (for envvar purposes)
def main():
    parser = argparse.ArgumentParser(description='Parse data from TCP log files')
    parser.add_argument('-i', '--input', type=str,
                    help='path from which to get input data', required=True)
    parser.add_argument('-d', '--dims', type=int, 
                    help='number of dimensions of input points', required=True)
    parser.add_argument('-c', '--compile', type=bool, nargs='?',
                    const=True, default=False, help="If set, compile all kmeans.")
    parser.add_argument('-s', '--specs', type=str, nargs='*', required=False,
                    help="Specify what specs to run (dump, opt)")
    
    args = parser.parse_args()
    infile = args.input
    dims = args.dims
    specs = args.specs

    # get all student directories
    kmeans = [name for name in os.listdir(".") if (os.path.isdir(os.path.join(os.getcwd(), name)) and "kmeans" in name)]

    # change cwd to git root
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.join(script_dir, "../../../"))
    print("cwd:", os.getcwd())

    signal.signal(signal.SIGINT, signal_handler)
    root_dir = os.path.join(os.getcwd(), "src/apps/uniform_kmeans")

    if args.compile:
        compile_kmeans(root_dir, kmeans)


    # generate dumps
    if "dump" in specs:
        ava = launch_ava_manager(4, "dump")
        sleep(3)
        print("running dump with root", root_dir, "kmeans", kmeans, "infile", infile, "dims", dims)
        run_kmeans_serial(os.path.join(os.getcwd(), root_dir), kmeans, infile, dims)
        sleep(3)
        kill_ava(ava, "dump")           

    # run all kmeans concurrently with opt
    if "opt" in specs:
        print("running opt with:", kmeans)
        ava = launch_ava_manager(4, "opt")
        sleep(3)
        run_kmeans_concurrent(root_dir, kmeans, infile, dims)
        sleep(2)
        kill_ava(ava, "opt")


if __name__ == "__main__":
    main()
