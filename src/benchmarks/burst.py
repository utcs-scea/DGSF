#!/usr/bin/env python3
import os, sys
from time import sleep
from subprocess import Popen, PIPE
from timeit import default_timer as timer
import numpy.random as rd
import threading
import pprint
import statistics
import json

BURST_COUNT = 10
SLEEP_BETWEEN = 2

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir( os.path.join(script_dir, "../..") )
apps_dir = os.path.join(script_dir, "../../src/apps")
apps_dir = os.path.abspath(apps_dir)

pending = 0
procs = []
pids = {}

def launch(job):
    global pending
    cmd = ["python3", "handler.py"]
    p = Popen(cmd, cwd=job[1], stdout=PIPE) #stderr=DEVNULL)
    pids[p.pid] = {}
    pids[p.pid]["name"] = job[0]
    pids[p.pid]["start"] = timer()
    procs.append(p)

done_launching = False
def wait():
    global pending, pids, procs
    while True:
        #all finished
        if pending == 0 and done_launching:
            print("please quit")
            return
        #create new list so we avoid iterating over done items
        new_procs = []
        #print("waiting.. on ", pending)
        #sleep(2)
        for p in procs:    
            # not done yet, check later
            if p.poll() is None:
                new_procs.append(p)
                continue
        
            pending -= 1
            print("done, pending: ", pending)
            print("child done, communicating")
            stdout, stderr = p.communicate()
            pids[p.pid]["elapsed"] = round(timer() - pids[p.pid]["start"], 2)
            del pids[p.pid]["start"]

            for line in stdout.decode().splitlines():
                if line.startswith("#$%"):
                    data = json.loads(line[3:])
                    print("Got: ", data)
                    pids[p.pid]["data"] = data

        procs = new_procs
        sleep(0.1)

bert = ("bert", os.path.join(apps_dir, "faas_bert"))
resnet = ("resnet", os.path.join(apps_dir, "faas_classification_detection"))
faceid = ("faceid", os.path.join(apps_dir, "faas_face_id"))
facedet = ("facedet", os.path.join(apps_dir, "faas_face_det"))
covid = ("covid", os.path.join(apps_dir, "covidct/refactored"))
kmeans = ("kmeans", os.path.join(apps_dir, "kmeans"))

workloads = [bert, resnet, faceid, facedet, covid, kmeans]

rd.seed(0)

pending = len(workloads) * BURST_COUNT
print("All pending ", pending)

#
# launch stuff
#
waiter = threading.Thread(target=wait)
waiter.start()
start = timer()

#launch workloads
for _ in range(BURST_COUNT):
    for i, job in enumerate(workloads):
        launch(job)
        print("launched ", i)

    sleep(SLEEP_BETWEEN)

done_launching = True
waiter.join()

end = timer()

pp = pprint.PrettyPrinter(indent=2)
#pp.pprint(pids)

print("Done, to process:\n", pp.pprint(pids))
# print(f"burst end-to-end, {end-start:.2f}\n")

# stats = {}
# jobnames = list(set([x[0] for x in jobs]))
# jobnames.sort()
# for j in jobnames:
#     runs = []
#     for _, v in pids.items():
#         if v["name"] == j:
#             runs.append(v["elapsed"])

#     print(f"{j},\t{statistics.mean(runs):.2f},\t{statistics.pstdev(runs) if len(runs) > 1 else 0:.2f}")
