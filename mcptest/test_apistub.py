import unittest
from typing import Dict, Any, List

class WeatherAggregator:
    """複数都市の天気情報を集約するクラス"""
    def __init__(self, api_client):
        self.api_client = api_client

    def get_multiple_cities_weather(self, cities: List[str]) -> Dict[str, Any]:
        """
        複数都市の天気情報を取得
        
        Args:
            cities: 都市名のリスト
            
        Returns:
            都市名をキーとした天気情報の辞書
        """
        result = {}
        for city in cities:
            result[city] = self.api_client.get_weather(city)
        return result


class WeatherAPIStub:
    def __init__(self):
        self.responses = {
            "Tokyo": {"temp": 25, "condition": "sunny"},
            "London": {"temp": 15, "condition": "cloudy"},
            "New York": {"temp": 20, "condition": "rainy"},
        }
        self.call_count = 0
        self.last_called_city = None

    def get_weather(self, city:str) -> Dict[str, Any]:          
        self.call_count += 1
        self.last_called_city = city

        if city in self.responses:
            return self.responses[city]
        else:
            return ValueError(f"City {city} not found")
        
class TestWithStub(unittest.TestCase):
    def test_weather_aggregator(self):
        print("------TestWithStub start------")
        stub = WeatherAPIStub()

        aggregator = WeatherAggregator(api_client=stub)

        cities = ["Tokyo", "London", "New York"]
        result = aggregator.get_multiple_cities_weather(cities)

        self.assertEqual(len(result), 3)
        self.assertEqual(result["Tokyo"]["temp"], 25)

            