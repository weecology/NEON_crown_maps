#test_lidar
from .. import LIDAR
import pytest

@pytest.fixture()
def laz_path():
    return "data/BART_041.laz"

def test_load_lidar(laz_path):
    pc = LIDAR.load_lidar(laz_path)
    
    #Has points
    assert not pc.data.points.empty
    