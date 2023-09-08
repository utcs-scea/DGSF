#!/usr/bin/env python3
import os, sys
from time import sleep
from subprocess import Popen, PIPE
from timeit import default_timer as timer

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir( os.path.join(script_dir, "../..") )

apps_dir = os.path.join(script_dir, "../../src/apps")
apps_dir = os.path.abspath(apps_dir)

print(f"Running script at {os.getcwd()}")

build = "release"
#each GPU has 15000
#resnet_mem = 13000
#bert_mem = 13000
resnet_mem = 8000
bert_mem = 5000

procs = []

def launch(job):
    cmd = ["python3", "handler.py"]
    p = Popen(cmd, cwd=job[1]) #, stdout=PIPE) #stderr=DEVNULL)
    procs.append(p)


def wait():
    for p in procs:
        p.communicate()

bert = ("bert", os.path.join(apps_dir, "faas_bert"))
resnet = ("resnet", os.path.join(apps_dir, "faas_classification_detection"))

start = timer()

#
# launch stuff
#
launch(bert)
launch(bert)
#sleep(3)
launch(resnet)
launch(resnet)


wait()

end = timer()

print(f"end-to-end, {end-start:.2f}")
