#!/usr/bin/env python3
import subprocess
import sys, os
from time import sleep
from timer import Timer

script_dir = os.path.dirname(os.path.realpath(__file__))

NGPUS = [1] 
NPROCS = [1, 2, 4, 8]
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
    cmd = f"src/common/tcp_timeline/server2 -n {nclients} -f baseline-10000000/{nclients}p-{ngpus}gpu-{trial}.log"
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
            for run in range(NRUNS): 
                finished = 0
                tcp = launch_tcp_server(nprocs, ngpus, run)
                print("Launched tcp server with", nprocs)
                while finished != nprocs:
                    sleep(1)

                    processes = []

                    for i in range(nprocs-finished):
                        cmd = og_cmd.copy()
                        cmd = [(a if a != "ID" else str(i)) for a in cmd]
                        p = subprocess.Popen(cmd, env=proc_env)
                        processes.append(p)

                    for p in processes:
                        _stdout, _stderr = p.communicate()
                        if p.returncode != 0:
                            print(f"!!!!!!! Process {p.pid} returned non-zero code {p.returncode}")
                        else:
                            finished = finished + 1

                sleep(1)
                tcp.terminate()
                tcp.communicate()


if __name__ == "__main__":
    main()
