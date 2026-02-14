import pytest
from datetime import datetime
from typing import List, Dict, Union, Optional

class TestDataFactory:
    @staticmethod
    def create_valid_data() -> Dict[str, Union[int, float, str]]:
        """
        Create a dictionary containing valid weather data.

        Returns:
            Dict[str, Union[int, float, str]]: A dictionary with keys such as 'temperature', 'humidity', 'pressure',
            'conditions', and 'timestamp', containing valid data.
        """
        return {
            "temperature": 22,
            "humidity": 65,
            "pressure": 1013.25,
            "conditions": "partly cloudy",
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_invalid_data() -> List[Dict[str, Optional[Union[int, float, str]]]]:
        """
        Create a list of dictionaries containing invalid weather data.

        Returns:
            List[Dict[str, Optional[Union[int, float, str]]]]: A list of dictionaries with invalid or edge-case data
            for testing purposes.
        """
        return [
            {"temperature": "twenty-five"},
            {"humidity": 50},
            {"temperature": -300},
            {"temperature": None, "humidity": None }
        ]
    
    @staticmethod
    def create_edge_case_data() -> List[Dict[str, Union[int, float]]]:
        """
        Create a list of dictionaries containing edge case weather data.

        Returns:
            List[Dict[str, Union[int, float]]]: A list of dictionaries with edge-case data such as extreme temperatures
            or humidity values.
        """
        return [
            {"temperature": -273.15, "humidity": 0},
            {"temperature": 100, "humidity": 100},
            {"temperature": 25.123456789}
        ]
    
    def test_multiple_data_cases(self) -> None:
        """
        Test multiple data cases to ensure temperature values are within a valid range.

        Validates:
            - The temperature in valid data is a number and within the range -273.15 to 1000.
            - The temperature in edge case data is within the range -273.15 to 1000.
        """
        valid_data = self.create_valid_data()
        assert isinstance(valid_data["temperature"], (int, float))
        assert  -273.15 <= valid_data["temperature"] <= 1000

        for edge_case in self.create_edge_case_data():
            temp = edge_case.get("temperature", None)
            if temp is not None:
                assert -273.15 <= temp <= 1000
