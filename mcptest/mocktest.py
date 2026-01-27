from unittest.mock import Mock, patch, MagicMock
import unittest
import httpx
from typing import Dict, Any

class ExternalAPIClient:
    def __init__(self):
        print("---init---")
        
    def get_weather_data(self, city: str) -> Dict[str, Any]:
        print("---get_weather_data---")
        response = httpx.get(f"https://api.weather.com/v1/{city}")
        return response

#class TestExternalAPIClient(unittest.TestCase):
@patch('httpx.get')
def test_get_weather_data_success(self, mock_get):
    mock_response = Mock()
    mock_response.json.return_value = {
            "city": "Tokyo",
            "temperature": 25,
            "conditions" : "sunny" 
    }
    mock_get.return_value = mock_response
    
if __name__ == "__main__":
    print("---Test Start---")
    client = ExternalAPIClient()
    ret = client.get_weather_data("Tokyo")
    print(ret)
    
    print("---Test Enc---")    
