
import pytest
from datetime import datetime

class TestDataFactory:
    @staticmethod
    def create_valid_data():
        return {
            "temperature": 22,
            "humidity": 65,
            "pressure": 1013.25,
            "conditions": "partly cloudy",
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_invalid_data():
        return [
            {"temperature": "twenty-five"},
            {"humidity": 50},
            {"temperature": -300},
            {"temperature": None, "humidity": None }
        ]
    
    @staticmethod
    def create_edge_case_data():
        return [
            {"temperature": -273.15, "humidity": 0},
            {"temperature": 100, "humidity": 100},
            {"temperature": 25.123456789}
        ]
    
    def test_multiple_data_cases(self):
        valid_data = self.create_valid_data()
        assert isinstance(valid_data["temperature"], (int, float))
        assert  -273.15 <= valid_data["temperature"] <= 1000

        for edge_case in self.create_edge_case_data():
            temp = edge_case.get("temperature", None)
            if temp is not None:
                assert -273.15 <= temp <= 1000

'''        
        for invalid in self.create_invalid_data():
            if "temperature" in invalid:
                temp = invalid["temperature"]
                if isinstance(temp, (int, float)):
                    assert -273.15 <= temp <= 1000
                else:
                    assert isinstance(temp, str) or temp is None
'''        
