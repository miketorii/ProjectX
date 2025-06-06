import contextlib
import os
import queue
import requests
import sys
import threading
import time

dirname = "/home/xxx/download/wordpress"

FILTERED = [".jpg",".gif",".png",".css"]
TARGET = "http://127.0.0.1:31337"
THREADS = 10

answers = queue.Queue()
web_paths = queue.Queue()

def gather_paths():
    print("----gather paths------")
    for root, _, files in os.walk('.'):
        for fname in files:
            if os.path.splitext(fname)[1] in FILTERED:
                continue
            path = os.path.join(root, fname)
            if path.startswith('.'):
                path = path[1:]
            print(path)
            web_paths.put(path)

def test_remote():
    while not web_paths.empty():
        path = web_paths.get()
        url = f'{TARGET}{path}'
        time.sleep(2)
        r = requests.get(url)
        if r.status_code == 200:
            answers.put(url)
            sys.stdout.write('+')
        else:
            sys.stdout.write('x')
        sys.stdout.flush()

def testprint():
    print("test print")
    
def run():
    print("-----run----")
    mythreads = list()
    for i in range(THREADS):
        print(f"Spawing thread {i}")
#        t = threading.Thread(target=testprint)
        t = threading.Thread(target=test_remote)
        mythreads.append(t)
        t.start()
    
    for thread in mythreads:
        thread.join()
    
@contextlib.contextmanager
def chdir(path):
    print("----chdir------")    
    this_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(this_dir)
    
if __name__ == "__main__":
    with chdir(dirname):
        gather_paths()
        
    input("Press return to continue.")
    run()
    with open('myanswers.txt','w') as f:
        while not answers.empty():
            f.write(f'{answers.get()}\n')        

    print("----done----")
