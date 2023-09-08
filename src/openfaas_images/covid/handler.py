import subprocess
import os, sys
from timeit import default_timer as timer
import boto3

s3_client = boto3.client(
    's3',
    aws_access_key_id="YOUR_aws_access_key_id",
    aws_secret_access_key="YOUR_aws_sercret_access_key_id"
)

#warm fatbins
flist = s3_client.list_objects(Bucket="hf-dgsf", Prefix="cuda_dumps/covid")['Contents']
#print(f"downloading {flist}", file=sys.stderr, flush=True)
if not os.path.exists("/tmp/cuda_dumps"):
    os.mkdir("/tmp/cuda_dumps")
    for key in flist:
        fname = key['Key'].split("/")[-1]
        #print("downloaded to ", "/tmp/cuda_dumps/"+fname, file=sys.stderr, flush=True)
        s3_client.download_file('hf-dgsf', key['Key'], "/tmp/cuda_dumps/"+fname)

s3_client.download_file('hf-dgsf', "covid/zoom_kernel.cubin", "/cuda_dumps/zoom_kernel.cubin")
print("Done downloading", file=sys.stderr, flush=True)

def handle(req):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    
    cmd = "python3 " + os.path.join(this_dir, "internal_handler.py")
    cmd = cmd.split()

    proc_env = os.environ.copy()
    proc_env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib"
    proc_env["AVA_GUEST_DUMP_DIR"] = "/tmp/cuda_dumps"
    proc_env["AVA_CONFIG_FILE_PATH"] = "/tmp/ava.conf"

    proc_env["TF_INTRA_OP_PARALLELISM_THREADS"] = "1"
    proc_env["TF_INTER_OP_PARALLELISM_THREADS"] = "1"
    
    # for local testing
    #proc_env["LD_LIBRARY_PATH"] = "/home/ubuntu/serverless-gpus/build/ava/release/onnx_opt/lib"
    proc_env["AVA_WORKER_DUMP_DIR"] = "/home/ubuntu/serverless-gpus/src/apps/covidct/cuda_dumps"

    print("Running cmd ", cmd, file=sys.stderr, flush=True)
    print("With env", proc_env, file=sys.stderr, flush=True)
    try:
        #p = subprocess.Popen(cmd, env=proc_env, cwd=this_dir, stdout=subprocess.PIPE)
        #stdout, _stderr = p.communicate()
        p = subprocess.Popen(cmd, env=proc_env, cwd=this_dir)
        p.communicate()

        print("finished", file=sys.stderr, flush=True)
        #if p.returncode != 0:
        #    print(f'*** Process {p.pid} returned non-zero code {p.returncode}', file=sys.stderr)
        #stdout = stdout.decode()
        #print(stdout, file=sys.stderr)
    except Exception as e:
        print("err ", e, file=sys.stderr, flush=True)

    ret = {
        "load_model": 0,
    }

    # for line in stdout.splitlines():
    #     print("! ", line)
    #     if "," not in line: continue
    #     k, v = line.split(",")
    #     ret[k] = v[:-3]

    print(ret, file=sys.stderr)
    return ret
