import paramiko

def ssh_command(ip, port, user, passwd, cmd):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, port=port, username=user, pkey=passwd)
    
if __name__ == "__main__":
    print("-----start-----")
    import getpass
    user = input('Username: ')
    password = getpass.getpass()

    ip = "127.0.0.1"
    port = 22
    cmd = "" #input('Enter command"')
    ssh_command(ip, port, user, password, cmd)
    print("---End---")
