import subprocess
import os, sys
from timeit import default_timer as timer
import boto3
import string, random

s3_client = boto3.client(
    's3',
    aws_access_key_id="YOUR_aws_access_key_id",
    aws_secret_access_key="YOUR_aws_sercret_access_key_id"
)

input_fname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

SCRIPT_NAME = "migration_bench"
EXTRA_ARGS = sys.argv[1]
DUMPS_PREFIX = "cuda_dumps/malloc"
DUMPS_DIR = "/home/ubuntu/serverless-gpus/src/apps/malloc/cuda_dumps"
HOST_AVA_LIBS = "/home/ubuntu/serverless-gpus/build/ava/release/onnx_opt/lib"
WORKER_DUMPS_DIR = "/home/ubuntu/serverless-gpus/src/apps/malloc/cuda_dumps"
REQ_MEMORY = "350"

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

# returns the dimension of points in the input file, required for argv
def get_dims(infile):
    fp = open(infile)
    dims = 0
    for i, line in enumerate(fp):
        if i == 1:
            dims = len(line.split()) - 1
    fp.close()
    return dims

def handle(req):
    this_dir = os.path.dirname(os.path.realpath(__file__))

    cmd = os.path.join(this_dir, SCRIPT_NAME) + " " + EXTRA_ARGS
    cmd = cmd.split()

    # generate env
    proc_env = os.environ.copy()
    #if native, we dont set anything
    if "RUN_NATIVE" in proc_env.keys():
        print("Running on native")
    else:
        print("Running on ava")
        # switch between running in open faas/aws or locally
        if not IN_CONTAINER:
            proc_env["LD_LIBRARY_PATH"] = HOST_AVA_LIBS
        else:
            proc_env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib"

        if not IN_CONTAINER:
            proc_env["AVA_CONFIG_FILE_PATH"] = "/home/ubuntu/serverless-gpus/tools/ava.conf"
        else:
            proc_env["AVA_CONFIG_FILE_PATH"] = "/tmp/ava.conf"
        proc_env["AVA_GUEST_DUMP_DIR"] = DUMPS_DIR
        proc_env["AVA_WORKER_DUMP_DIR"] = WORKER_DUMPS_DIR
        proc_env["AVA_REQUESTED_GPU_MEMORY"] = REQ_MEMORY

    #print("Running cmd ", cmd, file=sys.stderr, flush=True)
    #print("With env", proc_env, file=sys.stderr, flush=True)
    start = timer()
    try:
        p = subprocess.Popen(cmd, env=proc_env, cwd=this_dir )#, stdout=subprocess.PIPE)
        stdout, _stderr = p.communicate()
        #p = subprocess.Popen(cmd, env=proc_env, cwd=this_dir)
        #p.communicate()
        
        #print("finished", file=sys.stderr, flush=True)
        #if p.returncode != 0:
        #    print(f'*** Process {p.pid} returned non-zero code {p.returncode}')
        #stdout = stdout.decode()
        #print(stdout, file=sys.stderr)
    except Exception as e:
        print("err ", e, file=sys.stderr, flush=True)
    end = timer()

   
    print("segfaulting to exit..")
    os.kill(os.getpid(),11)

    return ret


if __name__ == '__main__':
    handle(None)