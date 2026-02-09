import pytest

class WeatherResponseParser:
    """天気APIのレスポンスをパースするクラス"""
    
    def parse_current(self, response):
        """現在の天気情報をパース"""
        print("---parse current---")
        data = response.get("data", {})
        current = data.get("current", {})
        return {
            "temperature": current.get("temperature"),
            "feels_like": current.get("feels_like"),
            "humidity": current.get("humidity"),
            "conditions": current.get("conditions")
        }
    
    def parse_location(self, response):
        """位置情報をパース"""
        print("---parse location---")
        data = response.get("data", {})
        location = data.get("location", {})
        return {
            "city": location.get("city"),
            "country": location.get("country"),
            "latitude": location.get("coordinates", {}).get("lat"),
            "longitude": location.get("coordinates", {}).get("lon")
        }
    
    def parse_forecast(self, response):
        """天気予報をパース"""
        print("---parse forecast---")
        data = response.get("data", {})
        forecast = data.get("forecast", [])
        return [
            {
                "date": item.get("date"),
                "high": item.get("high"),
                "low": item.get("low")
            }
            for item in forecast
        ]
    
@pytest.fixture
def sample_weather_response():
    return {
        "status" : "success",
        "data": {
            "location": {
                "city" :  "Tokyo",
                "country" : "Japan",
                "coordinates" : {
                    "lat": 35.6895,
                    "lon": 139.6917
                },
            },
            "current": {
                "temperature": 25.5,
                "feels_like": 27.2,
                "humidity": 60,
                "conditions": "sunny"
            },
            "forecast": [
                {"date": "2024-03-15", "high": 26, "low": 18},
                {"date": "2024-03-16", "high": 24, "low": 17}
            ]
        }
    }

@pytest.fixture
def weather_parser():
    return WeatherResponseParser()

def test_parse_current_conditions(sample_weather_response, weather_parser):
    current = weather_parser.parse_current(sample_weather_response)

    assert current["temperature"] == 25.5
    assert current["conditions"] == "sunny"
    assert "feels_like" in current