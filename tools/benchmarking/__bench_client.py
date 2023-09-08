#!/usr/bin/python3
import argparse
import os
import socket
import struct
import selectors
import signal
import sys
import types
import time
import multiprocessing  
from enum import IntEnum

TCP_IP = "0.0.0.0"
TCP_PORT = 50057

sel = selectors.DefaultSelector()

class Opcode(IntEnum):
    START = 0
    END = 1
    INIT = 2

def set_packet(cid, opcode):
    return struct.pack('>I', cid << 8 | int(opcode))

def send_op(socket, data, opcode):
    data.outb = set_packet(data.connid, opcode)
    socket.send(data.outb)

def service_connection(key):
    sock = key.fileobj
    data = key.data

    sent = send_op(sock, data, Opcode.START)

    # run things here
    if data.connid < 10:
        time.sleep(5)
    os.system("echo hello")
    print("id: ", data.connid)

    sent = send_op(sock, data, Opcode.END)

    sel.unregister(sock)
    sock.close()

def start_connections(num_conns):
    server_addr = (TCP_IP, TCP_PORT)
    for i in range(0, num_conns):
        connid = i + 1
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        sock.setblocking(True)
        sock.connect_ex(server_addr)
        events = selectors.EVENT_WRITE
        data = types.SimpleNamespace(connid=connid,
                                     outb=b'')
        sel.register(sock, events, data=data)

        if i == 0:
            data.outb = set_packet(num_conns, Opcode.INIT)
             # TODO low-probability race cond if this doesn't finish first? make blocking
            sock.send(data.outb) 
            sock.setblocking(False)
                               

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_conns")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, exit)

    start_connections(int(args.num_conns))

    handles = []

    try:
        events = sel.select(timeout=1)
        if events:
            for key, _ in events:
                p = multiprocessing.Process(
                    target=service_connection, args=(key,))
                p.start()
                handles.append(p)
            # else error? TODO
        for p in handles:
            p.join()
    except KeyboardInterrupt:
        print("caught keyboard interrupt, exiting")
    finally:
        sel.close()

if __name__ == '__main__':
    main()
