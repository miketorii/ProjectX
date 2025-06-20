import ipaddress
import os
import socket
import struct
import sys

from ctypes import *
import socket
import struct

class IP:
    def __init__(self, buff=None):
        print("-----init-----")
        header = struct.unpack('<BBHHHBBH4s4s', buff)
        self.ver = header[0] >> 4
        self.ihl = header[0] & 0xF
        
        self.tos = header[1]
        self.len = header[2]
        self.id = header[3]
        self.offset = header[4]
        self.ttl = header[5]
        self.protocol_num = header[6]
        self.sum = header[7]
        self.src = header[8]
        self.dst = header[9]
        
        self.src_address = ipaddress.ip_address(self.src)
        self.dst_address = ipaddress.ip_address(self.dst)        
        
        self.protocol_map = {1: "ICMP", 6: "TCP", 17: "UDP"}
        
        try:
            self.protocol = self.protocol_map[self.protocol_num]
        except Exception as e:
            print("%s No protocol for %s" % (e, self.protocol_num))
            self.protocol = str(self.protocol_num)
    
def sniff(host):
    print("--------sniff------")
    print(host)

    if os.name == 'nt':
        socket_protocol = socket.IPPROTO_IP
    else:
        socket_protocol = socket.IPPROTO_ICMP

    print("------- %d" % socket_protocol)
    sniffer = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket_protocol)
    sniffer.bind((host,0))
    sniffer.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)

    if os.name == 'nt':
        sniffer.ioctl(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)

    print("------before recvfrom-----")
    try:
        while True:
            raw_buffer = sniffer.recvfrom(65535)[0]
            ip_header = IP(raw_buffer[0:20])
            print(raw_buffer[0:20])
            print('Protocol: %s %s -> %s' % (ip_header.protocol, ip_header.src_address, ip_header.dst_address) )
    except KeyboardInterrupt:
        sys.exit()

if __name__ == "__main__":
    #host = "127.0.0.1"
    host = "10.0.0.4"    
    sniff(host)
