import ipaddress
import os
import socket
import struct
import sys

from ctypes import *
import socket
import struct

class IP(Structure):
    _fields_ = [
        ("ver", c_ubyte, 4),
        ("ihl", c_ubyte, 4),
        ("tos", c_ubyte, 8),
        ("len", c_ushort, 16),
        ("id", c_ushort, 16),
        ("offset", c_ushort, 16),        
        ("ttl", c_ubyte, 8),
        ("protocol_num", c_ubyte, 8),
        ("sum", c_ushort, 16),
        ("src", c_uint32, 32),
        ("dst", c_uint32, 32)        
    ]

    def ___new__(cls, socket_buffer=None):
        print("-----new-----")
        return cls.from_buffer_copy(socket_buffer)

    def __init__(self, socket_buffer=None):
        print("-----init-----")        
        self.src_address = socket.inet_ntoa( struct.pack("<L", self.src) )
        self.dst_address = socket.inet_ntoa( struct.pack("<L", self.dst) )        
        
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

    try:
        while True:
            raw_buffer = sniffer.recvfrom(65535)[0]
            ip_header = IP(raw_buffer[0:20])
            print('Protocol: %s %s -> %s' % (ip_header.protocol, ip_header.src_address, ip_header.dst_address) )
    except KeyboardInterrupt:
        sys.exit()

if __name__ == "__main__":
    host = "127.0.0.1"
    sniff(host)
