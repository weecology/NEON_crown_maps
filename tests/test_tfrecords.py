from .. import tfrecords
import tensorflow as tf
import pytest
import os

@pytest.fixture()
def patch_size():
    return 200
    
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
    iterator = tfrecords.create_dataset(filepath=record, batch_size=batch_size)
    output = iterator.get_next()
    
    #Check image shape
    with tf.Session() as sess:
        image = sess.run(output)
        assert image.shape == (batch_size, 800, 800, 3)
