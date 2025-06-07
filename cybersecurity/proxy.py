import sys
import socket
import threading

HEX_FILTER = ''.join(
    [(len(repr(chr(i)))==3) and chr(i) or '.' for i in range(256)]
)


def hexdump(src, length=16, show=True):
    print("--hexdump--")

def receive_from(connection):
    buffer = b""
    connection.settimeout(5)
    try:
        while True:
            data = connection.recv(4096)
            if not data:
                break

            buffer += data
    except:
        pass

    return buffer
            

def request_handler(buffer):
    return buffer

def response_handler(buffer):
    return buffer

def proxy_handler(client_socket, remote_host, remote_port, receive_first):
    print("---proxy handler----")
    remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    remote_socket.connect((remote_host, remote_port))

    if receive_first:
        remote_buffer = receive_from(remote_socket)
        if len(remote_buffer):
            print("[<==] Received %d bytes from remote" % len(remote_buffer))
            hexdump(remote_buffer)

            remote_buffer = response_handler(remote_buffer)
            client_socket.send(remote_buffer)
            print("[==>] Sent to local")

    
    
def server_loop(local_host, local_port, remote_host, remote_port, receive_first):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind((local_host, local_port))
    except Exception as e:
        print('bind error: %r' % e)
        sys.exit(0)

    print("[*] Listening on %s:%d" % (local_host, local_port))
    server.listen(5)
    while True:
        client_socket, addr = server.accept()
        line = "> Received incoming connectin from %s:%d" % (addr[0], addr[1])
        print(line)

        proxy_thread = threading.Thread(
            target=proxy_handler,
            args=(client_socket, remote_host, remote_port, receive_first)
        )
        proxy_thread.start()


def main():
    if len(sys.argv[1:]) !=5:
        print("proxy.py localhost localport remotehost remoteport receive_first")
        print("proxy.py 127.0.0.1 9000 10.12.132.1 9000 True")
        sys.exit(0)

    local_host = sys.argv[1]
    local_port = int(sys.argv[2])

    remote_host = sys.argv[3]
    remote_port = int(sys.argv[4])

    receive_first = sys.argv[5]

    if "True" in receive_first:
        receive_first = True
    else:
        receive_first = False

    print(f"arg: {local_host} {local_port} {remote_host} {remote_port} {receive_first}")    
    server_loop(local_host, local_port, remote_host, remote_port, receive_first)

if __name__ == "__main__":
    print("-------start-------")
    main()
    print("--------end--------")    
