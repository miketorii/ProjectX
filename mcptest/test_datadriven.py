import unittest
from parameterized import parameterized

def is_valid_temperature(celsius):
    return celsius >= -273.15

class TestTemperatureValidation(unittest.TestCase):
    
    @parameterized.expand([
        (-273.15, True, "absolute_zero"),
        (-273.16, False, "below_absolute_zero"),
        (0, True, "freezing_point"),
        (100, True, "boiling_point"),
    ])

    def test_parse_current_contions(self, celsius, expected, name):
        print("---testing---")
        result = is_valid_temperature(celsius)

        self.assertEqual(
            result,
            expected,
            f"Failed on case '{name}': input {celsius} should return {expected}"
        )
        
if __name__ == '__main__':
    unittest.main()
