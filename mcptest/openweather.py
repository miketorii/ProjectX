import os
import httpx
from typing import Any, Optional
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import asyncio

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
    print('---fetch weather data---')

    tmpdata={"id": 1, "name": "Tokyo",
             "weather": "Sunny",
             "description": "This is wether report",
             "main" : {
                 "temp":28,
                 "feels_like":10,
                 "temp_min":0,
                 "temp_max":100,
                 "humidity":68,
                 "pressure":200
             },
             "wind": "windy",
             "speed":"100",
             "deg":"50"}

    return tmpdata

##########################################
#
def format_weather_response(data: dict[str, Any]) -> str:
    print('-----format weather response------')

    city_name = data.get("name","none")

    main = data.get("main",{})
    temp = main.get("temp",0)
    feels_like = main.get("feels_like",0)
    temp_min = main.get("temp_min", 0)
    temp_max = main.get("temp_max", 0)    
    humidity = main.get("humidity",0)
    pressure = main.get("pressure",0)
    
    print(city_name)
    print(temp)
    print(feels_like)
    print(temp_min)
    print(temp_max)
    print(humidity)
    
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

    print('----get_weather---')
    
    if not city or not city.strip():
        return "エラー: 都市名を入力してください。"

    weather_data = await fetch_weather_data(city.strip())

    print(weather_data)
    
    if weather_data is None:
        return f"すみません。{city} の天気情報を取得できませんでした。都市名を確認してもう一度お試しください。"
    
    
    return format_weather_response(weather_data)

##################################
#
async def funcmain():
    city = 'Tokyo'
#    print(city.strip())
#    ret = await fetch_weather_data(city.strip())
    
    ret = await get_weather("Tokyo")
    print(ret)    

if __name__ == "__main__":
    print('-----main start----')
    asyncio.run(funcmain())
    
    #mcp.run()
    print('-----main end----')    
    
