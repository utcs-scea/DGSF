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
flist = s3_client.list_objects(Bucket='hf-dgsf', Prefix="cuda_dumps/kmeans")['Contents']
if not os.path.exists("/tmp/cuda_dumps"):
    os.mkdir("/tmp/cuda_dumps")
    for key in flist:
        fname = key['Key'].split("/")[-1]
        s3_client.download_file('hf-dgsf', key['Key'], "/tmp/cuda_dumps/"+fname)
        print("downloaded to ", "/tmp/cuda_dumps/"+fname)


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
    start = timer()
    s3_client.download_file('hf-dgsf', 'kmeans/1m_16d_16c.txt', '/tmp/1m_16d_16c.txt')
    end_download = timer()

    cmd = os.path.join(this_dir, "kmeans") + " --type raw --seed 123 -i /tmp/1m_16d_16c.txt"
    cmd = cmd.split()

    proc_env = os.environ.copy()
    proc_env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib"
    proc_env["AVA_GUEST_DUMP_DIR"] = "/tmp/cuda_dumps"
    proc_env["AVA_CONFIG_FILE_PATH"] = "/tmp/ava.conf"

    print("Running cmd ", cmd, file=sys.stderr, flush=True)
    print("With env", proc_env, file=sys.stderr, flush=True)
    try:
        p = subprocess.Popen(cmd, env=proc_env, cwd=this_dir, stdout=subprocess.PIPE)
        stdout, _stderr = p.communicate()
        #p = subprocess.Popen(cmd, env=proc_env, cwd=this_dir)
        #p.communicate()
        print("finished", file=sys.stderr, flush=True)
        #if p.returncode != 0:
        #    print(f'*** Process {p.pid} returned non-zero code {p.returncode}')
        stdout = stdout.decode()
        print(stdout, file=sys.stderr)
    except Exception as e:
        print("err ", e, file=sys.stderr, flush=True)

    os.remove("/tmp/1m_16d_16c.txt")

    ret = {
        "download_input": round((end_download-start)*1000, 2),
        "load_model": 0,
        #"end-to-end": total_time,
    }

    for line in stdout.splitlines():
        print("! ", line)
        if "," not in line: continue
        k, v = line.split(",")
        ret[k] = v[:-3]

    print(ret, file=sys.stderr)
    return ret
