#Test file for prediction methods
import pytest
from .. import predict
from .. import tfrecords

@pytest.fixture()
def patch_size():
    return 800

@pytest.fixture()
def record(patch_size):
    record = tfrecords.create_tfrecords(tile_path="data/OSBS_029.tif",patch_size=patch_size, savedir="output")    
    return record

def test_predict_tile(record):
    boxes = predict.predict_tile(record, batch_size=1)
    assert boxes.shape[1] == 6