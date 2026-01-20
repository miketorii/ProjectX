import os
import httpx
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

print('-----openweather----')

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = os.getenv("OPENWEATHER_BASE_URL")

print(API_KEY)
print(BASE_URL)

print('-----call FastMCP----')
# サーバーの作成
mcp = FastMCP("openweather")

@mcp.tool()
def hello(name: str) -> str:
    """Say hello to the user."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    print('-----main start----')
    mcp.run()
    print('-----main end----')    
    
