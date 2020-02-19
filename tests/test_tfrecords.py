import os
import sys
import tensorflow as tf
import pytest
from ..utils import tfrecords
import matplotlib
matplotlib.use("MacOSX")

@pytest.fixture()
def patch_size():
    return 300
    
@pytest.fixture()
def batch_size():
    return 2

@pytest.fixture()
def record(patch_size):
    record = tfrecords.create_tfrecords(tile_path="data/OSBS_029.tif",patch_size=patch_size, savedir="output")
    return record
    
def test_create_tfrecords(patch_size):
    written_records = tfrecords.create_tfrecords(tile_path="data/OSBS_029.tif",patch_size=patch_size, savedir="output")
    assert os.path.exists(written_records)

def test_create_dataset(record, batch_size):
    output = tfrecords.create_tensors(list_of_tfrecords=record, batch_size=batch_size)
    
    #Check image shape
    with tf.Session() as sess:
        image = sess.run(output)
        assert image.shape == (batch_size, 800, 800, 3)