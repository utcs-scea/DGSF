import socket
import sys
from time import sleep

ip = "127.0.0.1"
port = 50055 

# Create socket for server
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)

def socket_send(data):
    print("sending:", str(data))
    s.sendto(data.to_bytes(8, 'little'), (ip, port))

def socket_close():
    s.close()

# example
def main():
    for i in range(100):
        socket_send(i)
        sleep(.1)

if __name__ == "__main__":
    main()
