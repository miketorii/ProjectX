import httpx

import unittest
from unittest.mock import patch

def get_weather(city):
    response = httpx.get(f"https://api.example.com/{city}")
    return response.json()


class TestWeather(unittest.TestCase):

    @patch('httpx.get')
    def test_get_weather(self, mock_get):
        print("---test_get_weather---")

        mock_get.return_value.json.return_value = {"temp":25, "status":"Sunny"}

        result = get_weather("Tokyo")

        self.assertEqual(result["temp"], 25)

        mock_get.assert_called_once_with("https://api.example.com/Tokyo")

if __name__ == '__main__':
    unittest.main()
    
