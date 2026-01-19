import mcp
import asyncio
import time
from typing import Optional

#print(f'MCP version: {mcp.__version__}')

##############################################
#
class RateLimiter:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def aquire(self, tokens: int = 1) -> None:
        print('-------aquire-------')
        async with self.lock:
            while True:
                now = time.time()

                elapsed = now - self.last_update
                self.tokens = min(self.capacity, self.tokens+elapsed*self.rate)
                print(f'self.tokens = {self.tokens}')
                self.last_update = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                wait_time = (tokens - self.tokens) / self.rate
                print(f'wait_time = {wait_time}')
                await asyncio.sleep(wait_time)
                
##############################################
#
MAXRETRY = 4

RET_SUCESS = '200'
RET_ERROR = '404'

##############################################
#
def getdata(count):
    print(f'---get data--- {count}')
    ret = RET_ERROR

    if count == MAXRETRY-1:
        ret = RET_SUCESS

    print(f'---get data ret = {ret}')
    return ret

##############################################
#
async def request_func():
    print('---request func---')

    base_delay = 1.0
    
    retrycount = 0

    for i in range(MAXRETRY):    
        ret = getdata(retrycount)
        retrycount += 1

        delay = base_delay * (2 ** i)
        print(f'delay={delay}')

        await asyncio.sleep(delay)

##############################################
#
def getdata2():
    print('------get data 2------')
    
##############################################
#
async def make_rate_limited_request():
    print('-------make rate limited request----------')

    rate_limiter = RateLimiter(rate=2.0, capacity=10)

    await rate_limiter.aquire()

    getdata2()
    


##############################################
#    
if __name__ == '__main__':
    #asyncio.run(request_func())

    asyncio.run(make_rate_limited_request())
    
