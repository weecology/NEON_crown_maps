#Test file for prediction methods
import pytest
import glob
import platform
from dask.distributed import Client
from .. import predict
from .. import start_cluster

@pytest.fixture
def test_platform():
    test_platform = platform.system()
    return test_platform

@pytest.fixture()
def create_tile_list():
    tile_list = glob.glob("../data/*.tif")
    return tile_list

def test_predict_tiles(create_tile_list):
    boxes = predict.predict_tiles(create_tile_list, client=None)
    assert boxes.shape[1] == 6
        
def test_predict_tiles(create_tile_list, test_platform):
    #Skip locally, just test on cluster
    if test_platform =="Darwin":
        pass 
    else:
        client = start_cluster.start_dask_cluster(number_of_workers=2)
        boxes = predict.predict_tiles(create_tile_list, client=create_client)
        assert boxes.shape[1] == 6
    