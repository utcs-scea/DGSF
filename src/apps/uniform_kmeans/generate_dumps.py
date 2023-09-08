#!/usr/bin/env python3
import argparse
import sys
import os
import signal
import ctypes
import subprocess
from time import sleep

#this is the candidate list, which generation works
skip = [
    "kmeans-21",
    "kmeans-32",
    "kmeans-12",
    "kmeans-13",
    "kmeans-16",
    "kmeans-2",
    "kmeans-20",
    "kmeans-23",
    "kmeans-24",
    "kmeans-25",
    "kmeans-27",
    "kmeans-29",
    "kmeans-3",
    "kmeans-34",
    "kmeans-37",
    "kmeans-4",
    "kmeans-40",
    "kmeans-41",
    "kmeans-43",
    "kmeans-45",
    "kmeans-46",
    "kmeans-47",
    "kmeans-48",
    "kmeans-49",
    "kmeans-5",
    "kmeans-50",
    "kmeans-51",
    "kmeans-53",
    "kmeans-54",
    "kmeans-59",
    "kmeans-6",
    "kmeans-63",
    "kmeans-64",
    "kmeans-9",
]


def launch_process(cmd, wd):
    print(f"Launching: {cmd} at {wd}")
    subprocess.run(cmd, shell=True, cwd=wd)

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
        wd = os.path.join(base_dir, kmeans_dir)
        compile_cmd = get_cuda_cmd(wd, "How_To_Compile")
        print("> Compiling # ", kmeans_dir)
        print("> Compile command:", compile_cmd)
        launch_process(compile_cmd, wd)

def setup_run(script_dir, kmeans_dir, infile, dims):
    dump_dir = os.path.join(kmeans_dir, "cuda_dumps")
    infile_path = os.path.join(script_dir, infile)

    with open(os.path.join(kmeans_dir, "executable")) as f:
        exec_cmd = f.read().splitlines()[0]

    cmd = f"{exec_cmd} -k 16 -i {infile_path} -d {dims} -t 0.01 -m 200 -s 8675309"
    cmd = cmd.replace("\n", " ")
    print("> cmd:", cmd)
    print("> kmeans:", kmeans_dir)

    var_string = f"LD_LIBRARY_PATH={script_dir}/../../../build/ava/debug/onnx_dump/lib "
    var_string += f"AVA_CONFIG_FILE_PATH={script_dir}/../../../tools/ava.conf "

    cmd = var_string + cmd
    return (cmd, kmeans_dir, dump_dir)

def run_kmeans_serial(script_dir, kmeans, infile, dims):
    for kmeans_dir in kmeans:
        if kmeans_dir in skip:
            print(f"Skipping {kmeans_dir} because of skip list")
            continue

        print("launch", kmeans_dir)
        sleep(2)
        (cmd, wd, dump_dir) = setup_run(script_dir, os.path.join(script_dir, kmeans_dir), infile, dims)
        launch_process(cmd, wd)
        os.system(f"rm -rf {dump_dir}")
        os.system(f"mkdir {dump_dir}")
        os.system(f"cp /tmp/*.ava {dump_dir}")
        os.system(f"rm -f /tmp/*.ava")
        
        sleep(7)

# current assumption: this script is ran from the top directory (for envvar purposes)
def main():
    dims = 24
    infile = "inputs/small.txt"
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(os.path.join(script_dir, "../../../"))
    
    # get all student directories
    kmeans = [name for name in os.listdir(script_dir) if (os.path.isdir(os.path.join(script_dir, name)) and "kmeans" in name)]
    kmeans.sort()

    print(f"Running the following kmeans: {kmeans}")

    print("Compiling all")
    compile_kmeans(script_dir, kmeans)

    #generate dumps
    #print("running dump with root", script_dir, "kmeans", kmeans, "infile", infile, "dims", dims)
    #run_kmeans_serial(script_dir, kmeans, infile, dims)
    #sleep(3)


if __name__ == "__main__":
    main()
