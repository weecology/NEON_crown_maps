#Test file for prediction methods
import pytest
import glob
from distributed import Client, LocalCluster
from .. import predict

@pytest.fixture()
def create_dask_client():
    cluster = LocalCluster()
    return cluster

@pytest.fixture()
def create_tile_list():
    tile_list = glob.glob("../data/*.tif")
    return tile_list

def test_predict_tiles(create_tile_list, create_dask_client):
    predict.predict_tiles(create_tile_list, create_dask_client)
    
def test_predict_tiles_dask(create_tile_list, create_dask_client):
    predict.predict_tiles(create_tile_list, create_dask_client)
    