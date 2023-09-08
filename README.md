# serverless-gpus

This project uses Task, latest release. To install make sure you have Go setup and run:

```
go install github.com/go-task/task/v3/cmd/task@latest
```

# Other dependencies

We need rust to compile the webserver inside a VM:

`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
`rustup default nightly`

Redis server (for now), should be packaged in the repo

# Compiling and running

Run `task` to see a list of all tasks. Names should be enough to figure out what they do.
If you want to compile something but it says it's up-to-date, add the `-f` flag.
For example:

`task build-ava-release -f`


# Misc
CUDA version for ava 10.1:
wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run
wget https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run

cudnn:
Version 7.6.5 for 10.1 from:
https://developer.nvidia.com/rdp/cudnn-archive

need both library and developer
https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb
https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn7-dev_7.6.5.32-1+cuda10.1_amd64.deb


If the faas gateway is not playing nice, need to deploy with addr:

RESMNGR_ADDR=128.83.122.70  ./deploy_stack.sh