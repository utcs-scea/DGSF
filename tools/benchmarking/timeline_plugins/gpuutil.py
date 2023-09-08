from pynvml import *

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
util = nvmlDeviceGetUtilizationRates(handle)
util.gpu
util.memory


#requires python3 -m pip install nvidia-ml-py3
#and maybe python3 -m pip install nvidia-ml-py