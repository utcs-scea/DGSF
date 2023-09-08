import socket
import sys
import threading
import fcntl, os
from time import sleep
from copy import deepcopy

ip = "127.0.0.1"
port = 40057

received_data = []
lock = threading.Lock()

def listen_thread():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (ip, port)
    s.bind(server_address)
    fcntl.fcntl(s, fcntl.F_SETFL, os.O_NONBLOCK)
    #print("####### Server is listening #######")

    while True:
        lock.acquire()
        try:
            data = s.recv(128)
        except socket.error:
            pass
        else:
            #received_data.append(int.from_bytes(data, "little"))
            print(",,,,,MARK")
        lock.release()
        sleep(.1)

def socket_listen():
    t = threading.Thread(target=listen_thread)
    t.daemon = True
    t.start()

def socket_empty_data():
    lock.acquire()
    data = deepcopy(received_data)
    received_data.clear()
    lock.release()
    return data
