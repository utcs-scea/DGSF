#!/usr/bin/env python3
from pynvml import *
from time import sleep

import sys
sys.path.append("..")
from udp_server import socket_listen, socket_empty_data

INTERVAL_SEC = 0.2

def main():
    socket_listen()
    nvmlInit()

    devices = []
    deviceCount = nvmlDeviceGetCount()
    print("n devices: ", deviceCount)
    #   OVERRIDE
    #deviceCount = 2
    
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        devices.append(handle)

        ut, samp = nvmlDeviceGetEncoderUtilization ( handle) 

        print("sampling period (us) :", samp)


    headers = [f"gpu{i}" for i in range(deviceCount)]

    print(f"time, {','.join(headers)}, avg gpu, memory")
    time = 0
    while True:
        sum_util = .0
        sum_mem = .0
        #sum_power = .0
        each_util = []

        for i in range(deviceCount):
            ut = nvmlDeviceGetUtilizationRates(devices[i])
            sum_util += ut.gpu
            each_util.append(str(ut.gpu))

            #get used memory, and calc percentage
            #ut_mem = nvmlDeviceGetMemoryInfo(devices[i])
            #sum_mem += ut_mem.used / ut_mem.total
            ##sum_power += nvmlDeviceGetPowerUsage(devices[i])

        #data = socket_empty_data()
        #print("recv'ed data:", data)

        #average both
        #print(f"{time}, {','.join(each_util) },  {sum_util/deviceCount}, {(sum_mem/deviceCount)*100}")   #" , {sum_power/deviceCount}")
        print(f"{time:.2f}, {sum_util/deviceCount}")

        time += INTERVAL_SEC
        sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
