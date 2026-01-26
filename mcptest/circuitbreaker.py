import time

class SimpleCircuitBreaker:
    def __init__(self, fail_max=3, reset_timeout=10):
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSE"

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                print("--- State: HALF-OPEN---")
                print("--- Trying ---")
                self.state = "HALF-OPEN"
            else:
                raise Exception("Circuit is OPEN. Request blocked")
            
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        self.failure_count = 0
        self.state = "CLOSE"

    def _on_failure(self):
        self.failure_count +=1
        self.last_failure_time = time.time()
        if self.failure_count >= self.fail_max:
            self.state = "OPEN"
            print("------State: OPEN--------")
            print("------Start closing------")

def unstable_api():
    raise Exception("API Connection Error")

if __name__ == "__main__":
    print("-----Start--------")

    breaker = SimpleCircuitBreaker(fail_max=3, reset_timeout=5)
    
    for i in range(5):
        print(f"Try {i+1}")
        try:
            breaker.call(unstable_api)
        except Exception as e:
            print(e)
            
    print("-----End--------")    
