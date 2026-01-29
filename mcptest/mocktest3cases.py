from unittest.mock import Mock, patch, MagicMock
import unittest
import httpx
from typing import Dict, Any

class WeatherService:
    def __init__(self, api_client, cache, logger):
        print("---init---")
        self.api_client = api_client
        self.cache = cache
        self.logger = logger

    def get_weather_with_cache(self, city: str) -> Dict[str, Any]:
        cached_data = self.cache.get(f"weather:{city}")

        if cached_data:
            self.logger.info(f"Cache hit for {city}")
            return cached_data

        try:
            data = self.api_client.get_weather_date(city)
            self.cache.set(f"weather:{city}", data, ttl=3600)
            self.logger.info(f"fetched fresh data for {city}")
            return data        
        except Exception as e:
            self.logger.error(f"Error fetching weather for {city}: {e}")

class TestWeatherService(unittest.TestCase):
    def setUp(self):
        print('---TestWeatherService setUp---')
        self.mock_api_client = Mock()
        self.mock_cache = Mock()
        self.mock_logger = Mock()

        self.service = WeatherService(
            self.mock_api_client,
            self.mock_cache,
            self.mock_logger
        )
        
    def test_catch_hit(self):
        cached_data = {"city": "Tokyo", "temp":20}
        self.mock_cache.get.return_value = cached_data

        result = self.service.get_weather_with_cache("Tokyo")
        print(f'return value = {result}')

        self.mock_cache.get.assert_called_once_with("weather:Tokyo")
        self.mock_api_client.get_weather_data.assert_not_called()
        self.assertEqual(result, cached_data)
        self.mock_logger.info.assert_called_with("Cache hit for Tokyo")

    def test_cache_miss_and_api_success(self):
        self.mock_cache.get.return_value = None
        api_data = {"city": "Tokyo", "temp":20}
        self.mock_api_client.get_weather_data.return_value = api_data

        result = self.service.get_weather_with_cache("Tokyo")
        print(f'return value = {result}')

        self.mock_cache.get.assert_called_once_with("weather:Tokyo")
        self.mock_api_client.get_weather_data.assert_called_once_with("Tokyo")
        self.mock_cache.set.assert_called_once_with("weather:Tokyo", api_data, ttl=3600)
        self.assertEqual(result, api_data)

if __name__ == "__main__":
    print("---Test Start---")

    unittest.main()
    
    print("---Test Enc---")
    
