import win32api
import win32con
import win32security
import wmi

def get_process_privileges(pid):
    try:
        hproc = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION, False, pid)
        htok = win32security.OpenProcessToken(hproc, win32con.TOKEN_QUERY)
        privs = win32security.GetTokenInformation(htok, win32security.TokenPrivileges)
        privileges = ''
        for priv_id, flags in privs:
            if flags == win32security.SE_PRIVILEGE_ENABLED | win32security.SE_PRIVILEGE_ENABLED_BY_DEFAULT:
                privileges += f'{win32security.LookupPrivilegeName(None, priv_id)}|'        
    except Exception:
        privileges = 'N/A'

    return privileges

def log_to_file(message):
    with open('process_monitor_log.csv','a') as fd:
        fd.write(f'{message}\r\n')
        
def monitor():
    print("----------monitor-------------")
    head = ('CommandLine, Time, Executable, Parent PID, PID, User, Privileges')
    log_to_file(head)
    c = wmi.WMI()
    process_watcher = c.Win32_Process.watch_for('creation')
    print("process watcher")

    new_process = process_watcher()
    cmdline = new_process.CommandLine
    create_date = new_process.CreationDate
    executable = new_process.ExecutablePath
    parent_pid = new_process.ParentProcessId
    pid = new_process.ProcessId
    proc_owner = new_process.GetOwner()

    privileges = get_process_privileges(pid)
    process_log_message = (
        f'{cmdline}, {create_date}, {executable},'
        f'{parent_pid}, {pid}, {proc_owner}, {privileges}'
    )
    print(process_log_message)
    log_to_file(process_log_message)

            
#    while True:
#        try:

            
#       except Exception:
#                pass
    
    
if __name__ == "__main__":
    print("---------Start-------------")    
    monitor()
    print("---------End-------------")    
