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
    shapefile = predict.predict_tiles(model, [record], patch_size=400, raster_dir=["data"], save_dir="output", batch_size=model.config["batch_size"])
    
    return shapefile[0]

def test_load_lidar(laz_path):
    pc = LIDAR.load_lidar(laz_path)
    
    #Has points
    assert not pc.data.points.empty

def test_postprocess(laz_path, shapefile):
    #Load Lidar and Read Shapefile
    pc = LIDAR.load_lidar(laz_path)
    boxes = geopandas.read_file(shapefile)
    
    #Ensure numberic type
    boxes[["left","bottom","right","top"]] = boxes[["left","bottom","right","top"]].astype(float)    
    boxes = LIDAR.drape_boxes(boxes, pc)
    
    #assert there are remaining points l
    assert not boxes.empty
    
    #Generate new boxes
    assert all(boxes.columns.values == ["left","bottom","right","top","score","label","geometry","height"])
    