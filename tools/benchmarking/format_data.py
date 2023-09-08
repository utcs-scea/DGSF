#!/usr/bin/env python3
import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import json

procs = {}
baseprocs = {}

def format_data(proclist, datadir):
    for filename in os.listdir(datadir):
        if ".log" not in filename:
            print('log not in name, continuing')
            continue
        print("handle log:", filename)

        namesplit = filename.split('-')
        nprocs = int(namesplit[0].split('p')[0])
        ngpus = (namesplit[1].split('g')[0])

        nlines = 0
        with open(os.path.join(datadir, filename)) as f:
            for line in f:
                nlines = nlines + 1
                pass
            last_line = line

        if nlines != nprocs * 2:
            print("skipping over bad data file")
            continue

        time = int(last_line.split(':')[0])

        print("inserting", nprocs, "procs and", ngpus, "gpus")

        if nprocs not in proclist:
            proclist[nprocs] = {}
        if ngpus not in proclist[nprocs]:
            proclist[nprocs][ngpus] = []
        proclist[nprocs][ngpus].append(time)

    print(json.dumps(proclist, sort_keys=True, indent=4))

f, ax = plt.subplots(1)

def graph_data(proclist, baseproclist):
    print('keys:', proclist.keys())
    for proc in proclist.keys():
        gpus = sorted(proclist[proc].keys())
        print("plotting proc", proc)
        times = []
        for gpu in gpus:
            avg_time = mean(proclist[proc][gpu])
            times.append(avg_time / 1000)
        print('plotting times', times)

        basetime = mean(baseproclist[proc][str(1)]) / 1000

        line, = ax.plot(gpus, times, label = str(proc) + " processes (AvA)")
        ax.axhline(y=basetime, linestyle='--', label= str(proc) + " processes (baseline)", color=line.get_color())

def main():
    parser = argparse.ArgumentParser(description='Parse data from TCP log files')
    parser.add_argument('-r', '--rootdir', type=str, nargs=1,
                    help='path from which to get kmeans directories', required=True)
    parser.add_argument('-b', '--baselinedir', type=str, nargs=1,
                    help='path from which to get kmeans directories', required=True)
    
    args = parser.parse_args()
    root_dir = args.rootdir[0]
    base_dir = args.baselinedir[0]

    format_data(procs, root_dir)
    format_data(baseprocs, base_dir)

    graph_data(procs, baseprocs)

    ax.set_ylim(ymin=0)
    plt.legend()
    plt.title('Kmeans on AvA Runtimes')
    plt.xlabel('# GPUs')
    plt.ylabel('Time (s)')
    plt.show()
    plt.savefig('kmeans-graph.png')

if __name__ == "__main__":
    main()
