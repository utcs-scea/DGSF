#!/usr/bin/env python3
import os, sys
from time import sleep
from subprocess import Popen, PIPE, DEVNULL
from timeit import default_timer as timer
import numpy.random as rd
import threading
import pprint, json
import statistics
import numpy as np

LOW_LOAD = 3
HIGH_LOAD = 2

N_EACH_WORKLOAD = 10

# low load is
# LAMBDA = LOW
# WORKLOADS = "all"
# vary workers on the gpu server

LAMBDA = LOW_LOAD

# all or small
WORKLOADS = "all"
#WORKLOADS = "small"

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

#workloads = [covid, kmeans] #[bert, resnet, faceid, facedet, covid, kmeans]
workloads = []
if WORKLOADS == "all":
    workloads = [bert, resnet, faceid, facedet, covid, kmeans]
else:
    workloads = [bert, resnet, faceid, kmeans]

rd.seed(0)

jobs = []
for wl in workloads:
    jobs.extend([wl] * N_EACH_WORKLOAD)

rd.shuffle(jobs)
pending = len(jobs)
print("All workloads ", jobs)
print("All pending ", pending)

#intervals = rd.poisson(LAMBDA, (N_EACH_WORKLOAD*len(workloads)))
intervals = np.random.exponential(LAMBDA, size=(N_EACH_WORKLOAD*len(workloads)))

#
# launch stuff
#
waiter = threading.Thread(target=wait)
waiter.start()
start = timer()

for i, job in enumerate(jobs):
    #print("launching ", job)
    launch(job)

    if i == len(jobs)-1:
        break
    print("sleeping ", intervals[i])
    sleep(intervals[i])

done_launching = True
waiter.join()

end = timer()


#
# parse results
#

pp = pprint.PrettyPrinter(indent=2)
pp.pprint(pids)

total_times = []
stats = {}
jobnames = list(set([x[0] for x in jobs]))
jobnames.sort()

final_data = {}
for job in jobnames:
    final_data[job] = {"elapsed": [], "queue_time": []}
    for _, data in pids.items():
        if data["name"] == job:
            final_data[job]["elapsed"].append( data["elapsed"] )
            final_data[job]["queue_time"].append( data["data"]["queue_time"] )


#print("final data", final_data)

PRINT_ORDER = ["kmeans", "covid", "facedet", "faceid", "bert", "resnet"]


ordered_avg = []
ordered_std = []
ordered_q = []

executed_jobs = set([x[0] for x in workloads])
for job in PRINT_ORDER:
    if job not in executed_jobs:
        continue

    avg_elaps = statistics.mean  (final_data[job]["elapsed"])
    std_elaps = statistics.pstdev(final_data[job]["elapsed"]) if len(final_data[job]["elapsed"]) > 1 else 0
    avg_q = statistics.mean(final_data[job]["queue_time"])
    print(f"{job}, {avg_elaps:.2f}, {std_elaps:.2f}, {avg_q:.2f}")

    ordered_avg.append(f"{avg_elaps:.2f}")
    ordered_std.append(f"{std_elaps:.2f}")
    ordered_q.append(f"{avg_q:.2f}")

# for j in jobnames:
#     runs = []
#     for _, v in pids.items():
#         if v["name"] == j:
#             runs.append(v["elapsed"])
#             total_times.append(v["elapsed"])

#     print(f"{j},\t{statistics.mean(runs):.2f},\t{statistics.pstdev(runs) if len(runs) > 1 else 0:.2f}")

print("\n\n\n")
print(f"poisson end-to-end, {end-start:.2f}")
print(f"total computation, {sum(total_times):.2f}\n")

print("For pasting in graph source: ")
print(", ".join(ordered_avg))
print(", ".join(ordered_std))
print(", ".join(ordered_q))