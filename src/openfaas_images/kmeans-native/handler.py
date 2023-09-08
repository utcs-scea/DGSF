import subprocess
import os, sys
from timeit import default_timer as timer
import boto3

s3_client = boto3.client(
    's3',
    aws_access_key_id="YOUR_aws_access_key_id",
    aws_secret_access_key="YOUR_aws_sercret_access_key_id"
)

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

    print("Running cmd ", cmd, file=sys.stderr, flush=True)
    try:
        p = subprocess.Popen(cmd, cwd=this_dir, stdout=subprocess.PIPE)
        stdout, _stderr = p.communicate()
        print("finished", file=sys.stderr, flush=True)
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
