#!/usr/bin/env python3
from timer import Timer
from subprocess import run
from time import sleep
import sys


if len(sys.argv) < 3:
    print("need arguments: <n> <cmd> <arg> ...")
    sys.exit(1)

cmd = sys.argv[2:]
cmd = " ".join(cmd)
n = int(sys.argv[1])

print(f"Repeating {n} times (2s between each) the command: {cmd}")

for i in range(n):
    with Timer.get_handle("cmd"):
        run(cmd, shell=True)
    sleep(2)

print("\n\n")
Timer.print()