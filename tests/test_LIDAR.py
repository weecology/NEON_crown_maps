#test_lidar
import sys
import os
import geopandas

#relative path hack just for pytest
sys.path.append(os.path.dirname(os.getcwd()))

from .. import LIDAR
from .. import predict
from deepforest import deepforest
from ..utils import tfrecords
import pytest

@pytest.fixture()
def laz_path():
    return "data/OSBS_029.laz"


@pytest.fixture()
def record():
    written_records = tfrecords.create_tfrecords(tile_path="data/OSBS_029.tif",patch_size=400, savedir="output")
    return written_records

@pytest.fixture()
def shapefile(record):
    #Create model and set config
    model = deepforest.deepforest()
    model.use_release()
    
    #Predict
    #comet_experiment.log_parameters(model.config)
    shapefile = predict.predict_tiles(model, [record], patch_size=400, rgb_paths=["data/OSBS_029.tif"], save_dir="output", batch_size=model.config["batch_size"])
    
    return shapefile[0]

def test_postprocess_CHM(shapefile):
    draped_boxes = LIDAR.postprocess_CHM(shapefile, CHM="data/OSBS_029_CHM.tif", min_height=2)
    assert all(draped_boxes.columns.values == ["left","bottom","right","top","score","label","geometry","height","area"])