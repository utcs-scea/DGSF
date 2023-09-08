import re
import sys

memsets = []
memcpys = []
kernels = []

with open(sys.argv[1]) as f:
    for line in f:
        m = re.search(":::(\d+)", line)
        if m:
            #print(m[1])
            memsets.append(m[1])

        m = re.search(";;;(\d+)", line)
        if m:
            #print(m[1])
            memcpys.append(m[1])

        m = re.search("\?\?\?(\d+)", line)
        if m:
            #print(m[1])
            kernels.append(m[1])

#print(memsets)
#print("\n\n\n\n")
#print(memcpys)

#print(kernels)
print("\n".join(kernels))