import signal
import time
import sys

class GracefulKiller:
    def __init__(self):
        self.kill_now = False

        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)        

    def exit_gracefully(self, signum, frame):
        print(f"Got signal: {signum}")
        self.kill_now = True

def cleanup():
    print("---clean up---")
    time.sleep(2)
    print("terminate safely")
    sys.exit(0)
    
if __name__ == '__main__':
    print("-----Start-----")
    killer = GracefulKiller()

    while not killer.kill_now:
        print("---processing---")
        time.sleep(3)

    cleanup()

    print("-----End-----")    
