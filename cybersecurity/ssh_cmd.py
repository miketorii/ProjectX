import paramiko

key_path = "~/.ssh/id_rsa"

def ssh_command(ip, port, user, passwd, cmd):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    prv_key = paramiko.RSAKey.from_private_key_file(key_path)
    
#    client.connect(ip, port=port, username=user, pkey=passwd)
    client.connect(ip, port=port, username=user, pkey=prv_key)    

    _, stdout, stderr = client.exec_command(cmd)
    output = stdout.readlines() + stderr.readlines()
    if output:
        print("----output----")
        for line in output:
            print(line.strip())
            
if __name__ == "__main__":
    print("-----start-----")
    import getpass
    user = input('Username: ')
    password = getpass.getpass()

    ip = "127.0.0.1"
    port = 22
    cmd = "ls -al" #input('Enter command"')
    ssh_command(ip, port, user, password, cmd)
    print("---End---")
