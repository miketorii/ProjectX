import pytest
from parameterized import parameterized

class TestTemperatureValidation:
    
    @staticmethod
    def is_valid_temperature(celsius):
        return celsius >= -273.15
    
    @parameterized.expand([
        (-273.15, True, "absolute_zero"),
        (-273.16, False, "below_absolute_zero"),
        (0, True, "freezing_point"),
        (100, True, "boiling_point"),
    ])
    def test_is_valid_temperature(self, celsius, expected, name):
        print("---testing---")
        result = self.is_valid_temperature(celsius)
        assert result == expected, f"Failed on case {name}"
        
