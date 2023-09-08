#!/usr/bin/env python3
import argparse
import sys
import os
import signal
import ctypes
import subprocess
from time import sleep

kmeans_path = "src/apps/"
kmeans = [name for name in os.listdir(kmeans_path) \
        if (os.path.isdir(os.path.join(kmeans_path, name)) and "kmeans-" in name)]

uniform_path = "src/apps/uniform_kmeans"


for kmeans_dir in kmeans:
    path = os.path.join(kmeans_path, kmeans_dir)
    handler = uniform_path + "/" + "handler.py"
    launch = uniform_path + "/" + "launch_faas_kmeans.py"
    pycache = uniform_path + "/" + "__pycache__"
    inputs = uniform_path + "/" + "inputs"
    #os.system(f"cp {handler} {path}")
    #os.system(f"cp {launch} {path}")
    #os.system(f"cp -r {pycache} {path}")
    #os.system(f"cp -r {inputs} {path}")
    #print("cd into", path)
    #os.system(f"cd {path} && sed -i -- 's/uniform_kmeans/{kmeans_dir}/g' launch_faas_kmeans.py")
    #os.system(f"cd {path} && sed -i -- 's/uniform_kmeans/{kmeans_dir}/g' handler.py")
    #print("cwd:", os.getcwd())
    os.system(f"task serverless:build-app-image -- {kmeans_dir} --size 5000M")
 
sys.exit(0)

for kmeans_dir in kmeans:
    print("copying", kmeans_dir)
    original_path = os.path.join(kmeans_path, kmeans_dir)
    path = os.path.join("src/apps", kmeans_dir)
    try:
        os.mkdir(path)
    except:
        print("exists:", path)
    for i in range(50):
        copy_dir = "kmeans" + str(i)
        os.system(f"cp -r {original_path} {os.path.join(path, copy_dir)}")
