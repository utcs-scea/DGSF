#!/usr/bin/python3
import socket
import struct
import time
import argparse
import signal
import sys
import types
import selectors
from enum import IntEnum

TCP_IP = "0.0.0.0" #? 127.0.0.1 for localhost, emptystring for any
TCP_PORT = 50057   
RECORD = []

sel = selectors.DefaultSelector()

n_clients = 0
n_starts = 0
n_ends = 0

start_time = time.time()
end_time = time.time()

class Opcode(IntEnum):
    START = 0
    END = 1
    INIT = 2

def get_id(val):
    return (0xff00 & val) >> 8;

def get_opcode(val):
    return 0xff & val;

def accept_wrapper(sock):
    conn, addr = sock.accept()
    conn.setblocking(False)
    data = types.SimpleNamespace(addr=addr, utb=b'')
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel.register(conn, events, data=data)

def service_connection(key, mask):
    global n_clients
    global n_starts
    global n_ends
    global start_time
    global end_time

    sock = key.fileobj
    data = key.data

    if mask & selectors.EVENT_READ:
        recv_data = sock.recv(4)
        if recv_data:
            data = struct.unpack('>I', recv_data)[0]
            opcode = get_opcode(data)
            client_id = get_id(data)

            if opcode == Opcode.START:
                if n_starts == 0:
                    start_time = time.time()
                n_starts += 1
            elif opcode == Opcode.END:
                n_ends += 1
                if n_ends == n_clients:
                    end_time = time.time()
                    return end_time - start_time
            elif opcode == Opcode.INIT:
                n_clients = client_id
        else:
            sel.unregister(sock)
            sock.close()

    return 0

           
def tcp_listener():

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) #?
    sock.bind((TCP_IP, TCP_PORT))
    sock.listen()
    print('listening on', (TCP_IP, TCP_PORT))

    sock.setblocking(False)
    sel.register(sock, selectors.EVENT_READ, data=None)

    while True:
        events = sel.select(timeout=None)
        for key, mask in events:
            if key.data is None:
                accept_wrapper(key.fileobj)
            else:
                time = service_connection(key, mask)
                if time != 0:
                    print('time:', end_time - start_time)
                    break

def main():
    def sig_int(signal, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_int)
    tcp_listener()


if __name__ == '__main__':
    main()
