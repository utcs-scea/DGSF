#!/usr/bin/env python3
import subprocess
import sys, os
from time import sleep
from timer import Timer

script_dir = os.path.dirname(os.path.realpath(__file__))

NGPUS = [4, 3, 2, 1] 
NPROCS = [8, 4, 2, 1]
NRUNS = 11
NWARMUP = 0

#stuff to avoid orphans
import signal
import ctypes
libc = ctypes.CDLL("libc.so.6")
def set_pdeathsig(sig = signal.SIGTERM):
    def callable():
        return libc.prctl(1, sig)
    return callable

def launch_tcp_server(nclients, ngpus, trial):
    cmd = f"src/common/tcp_timeline/server2 -n {nclients} -f 10mil_logs/{nclients}p-{ngpus}gpu-{trial}.log"
    return subprocess.Popen(cmd.split(), preexec_fn=set_pdeathsig(signal.SIGTERM))

def launch_ava_manager(ngpus):
    cmd = f"""build/ava/release/svgpu_manager
        --worker_path build/ava/release/onnx_opt/bin/worker
        --manager_port 43200
        --worker_port_base 5100
        --worker_env LD_LIBRARY_PATH=/usr/local/cuda/nvvm/lib64
        --ngpus {ngpus}
        """
    return subprocess.Popen(cmd.split(), preexec_fn=set_pdeathsig(signal.SIGTERM))

def main():
    os.chdir(os.path.join(script_dir, "..", ".."))

    if len(sys.argv) < 2:
        print("Need at least command to run. Usage: K=V K2=V2 -cmd command to run")
        sys.exit(1)

    #read necessary env for the process we are spawning
    proc_env = os.environ.copy()
    for i, arg in enumerate(sys.argv[1:]):
        #everything after -cmd is the path to binary + args
        if arg == "-cmd":
            og_cmd = sys.argv[i+2:]
            break
        k,v = arg.split("=")
        proc_env[k] = v

    for ngpus in NGPUS:
        for nprocs in NPROCS:
            if ngpus == 4 and nprocs == 8:
                continue
            for run in range(NRUNS): 
                ava = launch_ava_manager(ngpus)
                sleep(2)
                print("Launched ava")
                
                tcp = launch_tcp_server(nprocs, ngpus, run)
                sleep(1)
                print("Launched tcp server")
                processes = []

                with Timer.get_handle(f"{ngpus}gpus_{nprocs}procs"):
                    for i in range(nprocs):
                        cmd = og_cmd.copy()
                        cmd = [(a if a != "ID" else str(i)) for a in cmd]
                        p = subprocess.Popen(cmd, env=proc_env)
                        processes.append(p)

                    for p in processes:
                        _stdout, _stderr = p.communicate()
                        if p.returncode != 0:
                            print(f"!!!!!!! Process {p.pid} returned non-zero code {p.returncode}")

                sleep(1)
                tcp.terminate()
                tcp.communicate()

                ava.terminate()
                ava.communicate()
                print("*** KILLED AVA***")
                sleep(1)
                subprocess.call("pkill -f home/eyoon/serverless-gpus/build/ava/release/onnx_opt/bin/worker", shell=True)
                sleep(3)

    Timer.print(warmups=NWARMUP)

if __name__ == "__main__":
    main()
