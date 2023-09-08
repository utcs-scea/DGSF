#!/usr/bin/env python3
import subprocess
import sys
from timer import Timer

def main():
    if len(sys.argv) < 3:
        print("Need at least 2 arguments: <# of processes> <command to launch>.")
        print("Any argument that is equal to ID will be replaced with the process id 0<=id<n")
        sys.exit(1)

    n = int(sys.argv[1])
    og_cmd = sys.argv[2:]

    with Timer.get_handle("end-to-end"):
        processes = []
        for i in range(n):
            cmd = og_cmd.copy()
            cmd = [(a if a != "ID" else str(i)) for a in cmd]
            p = subprocess.Popen(cmd)
            processes.append(p)
            print("launched proc", i, "with command", cmd)

        #if we need stdout or whatever:
        # https://docs.python.org/3/library/subprocess.html#subprocess.Popen.communicate
        for p in processes:
            _stdout, _stderr = p.communicate()
            if p.returncode != 0:
                print(f"!!!!!!! Process {p.pid} returned non-zero code {p.returncode}")
    
    Timer.print()

if __name__ == "__main__":
    main()
