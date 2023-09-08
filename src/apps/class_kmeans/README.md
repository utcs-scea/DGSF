## K-means structure:
each kmeans directory has the pre-compiled executable, a file called 'executable' that has the command to run, and a cuda_dumps directory for fatbins

for inputs/small.txt, dims (-d in all these scripts) is 24
for inputs/1000000-10.txt, dims is 10

## Scripts:

run_class_kmeans.py
- Can do a few things:
    - generate dumps for all kmeans (specify `-s dump`)
    - run opt with all kmeans on ava, good for debugging (specigy `-s opt`)
    - re-compile all kmeans (-c), would NOT recommend this because a lot of them fail (just keep the existing binaries)

launch_faas_kmeans.py
- Runs all kmeans on serverless with either no latency between runs (default) or launched based on a poisson distribution (specify `-p`)
- Will deploy, sleep for #kmeans * 2 seconds, then launch as a POST requeset in a thread per kmeans
- Prints end-to-end time to stdout

concurrent_baseline.py
- Runs all class kmeans on the host concurrently (launched with 1 second in-between each) up to n times. For each run, writes end-to-end time to a file "concurrent-baseline.txt". Will cycle through assigning gpus based on the --ngpus argument.

serial_baseline.py
- Splits kmeans into 4 groups, one per GPU, and creates a thread per GPU that runs each of its kmeans serially. For each run (specified through --nruns flag), writes end-to-end time to "serial-baseline.txt"

host_stats.py
- Calculates mean, stddev, min, and max runtimes for kmeans on the host by running all kmeans serially.

faas_stats.py
- Calculates mean, stddev, min, and max runtimes for kmeans in serverless by running all kmeans serially. Need to run with the resmngr, fn-server, and GPU server in different panes.


