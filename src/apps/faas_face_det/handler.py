import subprocess
import os, sys
from timeit import default_timer as timer
import boto3

s3_client = boto3.client(
    's3',
    aws_access_key_id="YOUR_aws_access_key_id",
    aws_secret_access_key="YOUR_aws_sercret_access_key_id"
)

SCRIPT_NAME = "detect.py"
EXTRA_ARGS = "--pic_fmt jpg --bsize 16 --gpu"
DUMPS_PREFIX = "cuda_dumps/onnxrt"
DUMPS_DIR = "/tmp/onnxrt/cuda_dumps"
HOST_AVA_LIBS = "/home/ubuntu/serverless-gpus/build/ava/release/onnx_opt/lib"
WORKER_DUMPS_DIR = "/home/ubuntu/serverless-gpus/src/apps/mlperf/cuda_dumps/onnxruntime/"
REQ_MEMORY = "12000"
os.environ["FRAMEWORK"] = 'onnxrt'

IN_CONTAINER = "OPENFAAS" in os.environ.keys() or "AWS_LAMBDA_RUNTIME_API" in os.environ.keys()

def download_dumps(prefix):
    flist = s3_client.list_objects(Bucket="hf-dgsf", Prefix=prefix)['Contents']
    def download_obj(obj):
        fname = obj.split("/")[-1]        
        s3_client.download_file('hf-dgsf', obj, os.path.join(DUMPS_DIR, fname))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as executor:
        for obj in flist:
            executor.submit(download_obj, obj["Key"])

if "RUN_NATIVE" not in os.environ.keys():
    if not os.path.exists(DUMPS_DIR):
        os.makedirs(DUMPS_DIR)
        download_dumps(DUMPS_PREFIX)
print("Done downloading dumps", file=sys.stderr, flush=True)


def handle(req):
    global EXTRA_ARGS
    this_dir = os.path.dirname(os.path.realpath(__file__))
    
    # generate env
    proc_env = os.environ.copy()
    if "RUN_CPU" in proc_env.keys():
        EXTRA_ARGS += " --cpu_threads 6"
    #if native, we dont set anything
    elif "RUN_NATIVE" in proc_env.keys():
        EXTRA_ARGS += " --gpu"
        print("Running on native")
    else:
        EXTRA_ARGS += " --gpu"
        print("Running on ava")
        # switch between running in open faas or locally
        if not IN_CONTAINER:
            proc_env["LD_LIBRARY_PATH"] = HOST_AVA_LIBS
        else:
            proc_env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib"

        if not IN_CONTAINER:
            proc_env["AVA_CONFIG_FILE_PATH"] = "/home/ubuntu/serverless-gpus/tools/ava.conf"
        else:
            proc_env["AVA_CONFIG_FILE_PATH"] = "/home/app/ava.conf"
        proc_env["AVA_GUEST_DUMP_DIR"] = DUMPS_DIR
        proc_env["AVA_WORKER_DUMP_DIR"] = WORKER_DUMPS_DIR
        proc_env["AVA_REQUESTED_GPU_MEMORY"] = REQ_MEMORY
    
    #generate cmd
    cmd = "python3 " + os.path.join(this_dir, SCRIPT_NAME) + " " + EXTRA_ARGS
    cmd = cmd.split()

    #print("Running cmd ", cmd, file=sys.stderr, flush=True)
    #print("With env", proc_env, file=sys.stderr, flush=True)
    start = timer()
    try:
        p = subprocess.Popen(cmd, env=proc_env, cwd=this_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        #print("finished", file=sys.stderr, flush=True)
        if p.returncode != 0:
            print(f'*** Process {p.pid} returned non-zero code {p.returncode}', file=sys.stderr)
        stdout = stdout.decode()
        #print(f"std: !!!\n{stdout}\n!!!")
        #print(f"err: !!!\n{stderr.decode()}\n!!!")
    except Exception as e:
        print("err ", e, file=sys.stderr, flush=True)
    end = timer()

    import json
    ret = {}
    qtime = None
    for line in stdout.splitlines(): 
        print("line: ", line)
        if line.startswith(">!!"):
            line = line[3:]
            ret = json.loads(line)
        if line.startswith(">!>"):
            k, v = line.split(",")
            qtime = v

    if qtime is not None:
        ret["queue_time"] = round(float(qtime), 4)
    ret["end-to-end-outside"] = round((end-start)*1000, 2)
    
    print("#$%"+json.dumps(ret), flush=True)
    return ret

if __name__ == '__main__':
    handle(None)
