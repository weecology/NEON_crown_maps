#Test file for prediction methods
import os
import sys
#relative path hack just for pytest
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import pytest
import glob
from deepforest import deepforest
from ..utils import tfrecords
from .. import predict

@pytest.fixture()
def model():
    model = deepforest.deepforest()
    model.use_release()
    return model

@pytest.fixture()
def patch_size():
    return 300

@pytest.fixture()
def record(patch_size):
    record = tfrecords.create_tfrecords(tile_path="data/OSBS_029.tif",patch_size=patch_size, savedir="output")    
    return record

@pytest.fixture()
def record_list(patch_size):
    tifs = glob.glob("data/*.tif")
    record_list = [ ]
    for tif in tifs:
        record = tfrecords.create_tfrecords(tile_path=tif,patch_size=patch_size, savedir="output")    
        record_list.append(record)
        
    return record_list

def test_predict_tile(model, record, patch_size):
    boxes = predict.predict_tile(model, record, patch_size=patch_size, batch_size=2)
    assert (boxes.columns == ['xmin', 'ymin', 'xmax', 'ymax', 'score', 'label',"filename"]).all()
    
def test_predict_tilelist(model, record_list,patch_size):
    boxes = predict.predict_tiles(model, records=record_list,patch_size=patch_size, batch_size=2,raster_dir=["data","data"], score_threshold=0.05,max_detections=300,classes={0:"Tree"},save_dir="output")    
    assert len(boxes) == len(record_list)
    assert os.path.exists("output/OSBS_029.shp")
