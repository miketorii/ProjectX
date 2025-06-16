import getpass
import os
import socket
import select
import sys
import threading
import argparse

import paramiko

SSH_PORT = 22
DEFAULT_PORT = 4000

g_verbose = True

key_path = "~/.ssh/id_rsa"

def verbose(s):
    if g_verbose:
        print(s)

######################################################
#
#
def handler(chan, host, port):
    print("---------handler----------")
    sock = socket.socket()
    try:
        sock.connect((host, port))
    except Exception as e:
        verbose("Forwarding request to %s:%d failed: %r" % (host, port, e))
        return

    verbose(
        "Connected! Tunnel open %r -> %r -> %r"
        % (chan.orgin_addr, chan.getpeername(), (host, port))
    )

    while True:
        r, w, x = select.select([sock, chan],[],[])
        if sock in r:
            data = sock.recv(1024)
            if len(data) == 0:
                break
            chan.send(data)
        if chan in r:
            data = chan.recv(1024)
            if len(data) == 0:
                break
            sock.send(data)

    chan.close()
    sock.close()
    verbose("Tunnel closed from %r" % (chan.orgin_addr,) )

def reverse_forward_tunnel(server_port, remote_host, remote_port, transport):
    print("-----------reverse forward tunnel---------")
    transport.request_port_forward("", server_port)
    
    while True:
        chan = transport.accept(1000)
        if chan is None:
            continue

        thr = threading.Thread(
            target=handler, args=(chan, remote_host, remote_port)
        )
        thr.setDaemon(True)
        thr.start()




######################################################
#
#        
def main():
    server_ip = "127.0.0.1"
    server_port = 22
    user = input('Username: ')

    option_port = 80
    
    remote_ip = "127.0.0.1"
    remote_port = 2000
    
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    prv_key = paramiko.RSAKey.from_private_key_file(key_path)

    verbose("Connecting to ssh host %s:%d ..." % (server_ip, server_port) )
    
    client.connect(server_ip, port=server_port, username=user, pkey=prv_key)    

    verbose("Now forwading remote port %d to %s:%d ..." % (option_port, remote_ip, remote_port) )

    try:
        reverse_forward_tunnel(
            option_port, remote_ip, remote_port, client.get_transport()
        )
        
    except KeyboardInterrupt:
        print("C-c: Port forwarding stopped.")
        sys.exit(0)
    
    
if __name__ == "__main__":
    print("--------start----------")
    main()
    print("--------end----------")    
    
