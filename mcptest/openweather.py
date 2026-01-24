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

    url = f"{BASE_URL}"
    params = {
        "lat": 35.6895,
        "lon": 139.6917,
        "appid": API_KEY,
        "units": "metric"
    }

    print(url)
    print(params)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                print(f'City not found: {city}')
            elif e.response.status_code == 401:
                print(f'Invalid API key')
            else:
                print(f'HTTP error: {e}')
            return None
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return None

##########################################
#
async def fetch_weather_data_test(city: str) -> Optional[dict[str, Any]]:
    print('---fetch weather data test---')

    url = f"{BASE_URL}/weather"
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "ja"
    }

    print(url)
    print(params)

    tmpdata={"id": 1,
             "name": "Tokyo",
             "sys": {
                 "country": "JA"
             },
             "weather": [
                 {
                     "description": "This is wether report"
                 }
             ],
             "main" : {
                 "temp":28,
                 "feels_like":10,
                 "temp_min":0,
                 "temp_max":100,
                 "humidity":68,
                 "pressure":200
             },
             "wind":{
                 "speed":100,
                 "deg":50 }
             }

    return tmpdata

##########################################
#
def format_weather_response(data: dict[str, Any]) -> str:
    print('-----format weather response------')

    city_name = data.get("name","none")

    sys = data.get("sys",{})
    country = sys.get("country","")

    print(sys)
    print(country)
    
    weather = data.get("weather", [{}])[0]
    description = weather.get("description","不明")

    print(weather)
    print(description)
    
    main = data.get("main",{})
    temp = main.get("temp",0)
    feels_like = main.get("feels_like",0)
    temp_min = main.get("temp_min", 0)
    temp_max = main.get("temp_max", 0)    
    humidity = main.get("humidity",0)
    pressure = main.get("pressure",0)

    print(main)
    print(city_name)
    print(temp)
    print(feels_like)
    print(temp_min)
    print(temp_max)
    print(humidity)
    print(pressure)    

    wind = data.get("wind", {})
    wind_speed = wind.get("speed",0)
    wind_deg = wind.get("deg",0)

    print(wind)
    print(wind_speed)
    print(wind_deg)

    wind_direction = convert_degrees_to_direction(wind_deg)

    response = f"""
    {city_name}, {country} の現在の天気

    天候: {description}
    現在の気温: {temp:.1f} C
    最低/最高気温{temp_min:.1f} C/ {temp_max:.1f} C
    湿度: {humidity}%
    気圧: {pressure} hPa
    風: {wind_direction} {wind_speed} m/s
    """
    
    return response.strip()

##########################################
#
def convert_degrees_to_direction(degrees: float) -> str:
    directions = [
        "北","北北東","北東","東北東","東","東南東","南東","南南東",
        "南","南南西","南西","西南西","西","西北西","北西","北北西"        
    ]
    
    index = round(degrees /22.5) % 16
    return directions[index]

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
    
