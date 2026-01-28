from unittest.mock import Mock, patch, MagicMock
import unittest
import httpx
from typing import Dict, Any

class ExternalAPIClient:
    def __init__(self):
        print("---init---")
        
    def get_weather_data(self, city: str) -> Dict[str, Any]:
        print("---IN: get_weather_data---")
        response = httpx.get(f"https://api.weather.com/v1/{city}")
        print("---OUT: get_weather_data---")
        return response.json()

class TestExternalAPIClient(unittest.TestCase):
    @patch('httpx.get')
    def test_get_weather_data_success(self, mock_get):
        print('---test_get_weather_data_success---')
        mock_response = Mock()
        mock_response.json.return_value = {
            "city": "Tokyo",
            "temperature": 25,
            "conditions" : "sunny" 
        }
        mock_get.return_value = mock_response

        client = ExternalAPIClient()
        ret = client.get_weather_data("Tokyo")
        print(f'return value = {ret}')
        
        self.assertEqual(ret["city"], "Tokyo")
        self.assertEqual(ret["temperature"], 25)
#        self.assertEqual(ret["temperature"], 30)        
        
if __name__ == "__main__":
    print("---Test Start---")

    unittest.main()
    
    print("---Test Enc---")    
