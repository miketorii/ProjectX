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
print(f'Using API base URL={BASE_URL}')

print('-----call FastMCP----')
# サーバーの作成
mcp = FastMCP("openweather")

##########################################
#
async def fetch_weather_data(city: str) -> Optional[dict[str, Any]]:
    return None

##########################################
#
def format_weather_response(data: dict[str, Any]) -> str:
    return "not implemented"

##########################################
#
@mcp.tool()
async def get_weather(city: str) -> str:
    """
    指定された都市の現在の天気を取得します

    Args:
        city: 都市名 (例: Tokyo, London, New York)

    Returns:
        天気情報のフォーマットされた文字列
    """

    if not city or not city.strip():
        return "エラー: 都市名を入力してください。"

    weather_data = await fetch_weather_data(city.strip())

    if weather_data is None:
        return f"すみません。{city} の天気情報を取得できませんでした。都市名を確認してもう一度お試しください。"
    
    
    return format_weather_response(weather_data)

if __name__ == "__main__":
    print('-----main start----')
    ret = get_weather("Tokyo")
    print(ret)
    
    mcp.run()
    print('-----main end----')    
    
