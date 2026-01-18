import mcp
import asyncio

#print(f'MCP version: {mcp.__version__}')

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
        
if __name__ == '__main__':
    asyncio.run(request_func())
    
