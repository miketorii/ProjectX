import pytest

@pytest.fixture
def sample_user():
    return {"id":1, "name": "Mike"}

def test_user_name(sample_user):
    assert sample_user["name"] == "Mike"

def test_user_id(sample_user):
    assert sample_user["id"] == 1

