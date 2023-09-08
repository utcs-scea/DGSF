import subprocess
import os
from timeit import default_timer as timer

#turns out we can't assume what we is in this dir can be accessed with "." because it isnt.

# returns the dimension of points in the input file, required for argv
def get_dims(infile):
    fp = open(infile)
    dims = 0
    for i, line in enumerate(fp):
        if i == 1:
            dims = len(line.split()) - 1
    fp.close()
    return dims

def handle(event, _):

    this_dir = os.path.dirname(os.path.realpath(__file__))
    start = timer()

    local_input = os.path.join(this_dir, "inputs/1000000p-10d.txt")
    cmd = [os.path.join(this_dir, "kmeans"), '-k', '16', '-m', '2000', '-t', '0', '-d', '10', '-i', local_input, '-r', '1', '-bench', '-s', '8675309', '-g']
    print(f"running {cmd}")
    proc_env = os.environ.copy()
    p = subprocess.Popen(cmd, env=proc_env, cwd=this_dir)
    _stdout, _stderr = p.communicate()
    if p.returncode != 0:
        print(f'*** Process {p.pid} returned non-zero code {p.returncode}')

    end = timer()
    total_time = end - start
    print("total:", end-start)
    return {'result': total_time}

if __name__ == "__main__":
    handle(None, None)
