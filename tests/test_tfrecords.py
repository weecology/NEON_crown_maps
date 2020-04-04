import matplotlib
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt

import os
import sys
import tensorflow as tf
import pytest
import numpy as np
from PIL import Image
import cv2
from skimage import measure

from ..utils import tfrecords
from deepforest import preprocess
from keras_retinanet.utils import image as keras_retinanet_image

@pytest.fixture()
def patch_size():
    return 300
    
@pytest.fixture()
def batch_size():
    return 2

@pytest.fixture()
def tile_path():
    return "data/OSBS_029.tif"

@pytest.fixture()
def record(tile_path, patch_size):
    record = tfrecords.create_tfrecords(tile_path=tile_path, patch_size=patch_size, patch_overlap=0.05, savedir="output")
    return record
    
def test_create_tfrecords(tile_path, patch_size):
    written_records = tfrecords.create_tfrecords(tile_path=tile_path, patch_size=patch_size, savedir="output")
    assert os.path.exists(written_records)

def test_create_dataset(tile_path,record, batch_size,patch_size):
    output = tfrecords.create_tensors(list_of_tfrecords=record, batch_size=batch_size)
    
    #Check image shape
    with tf.Session() as sess:
        tf_batch = sess.run(output)
        assert tf_batch.shape == (batch_size, 800, 800, 3)
    
    #match it to the same image from deepforest preprocess.
    raster = Image.open(tile_path)
    numpy_image = np.array(raster)        

    #Compute sliding window index
    windows = preprocess.compute_windows(numpy_image, patch_size, patch_overlap=0.05)

    #Save images to tmpdir
    predicted_boxes = []

    crop = numpy_image[windows[0].indices()] 

    #Crop is RGB channel order, change to BGR
    crop = crop[...,::-1]
    
    deepforest_image        = keras_retinanet_image.preprocess_image(crop)
    deepforest_image, scale = keras_retinanet_image.resize_image(deepforest_image)
    
    #assert that the first image is the same
    tf_image = tf_batch[0,:,:,:]
    
    if not np.array_equal(deepforest_image,tf_image):
        fig,axes = plt.subplots(1,3)
        axes = axes.flatten()
        
        axes[0].imshow(deepforest_image)
        axes[0].set_title("DeepForest")
        
        axes[1].imshow(tf_image)
        axes[1].set_title("Tfrecords")
        
        score, diff = measure.compare_ssim(deepforest_image,tf_image,multichannel=True, full=True)
        print("SSIM simarlity score {}".format(score))
        axes[2].imshow(diff)
        axes[2].set_title("Image difference")
        plt.show()

def test_yields_all_pngs():
    """On hipergator there is a rare error where some .png are skipped, using a sample tile that erred. Due to large file size this can only be run locally as debug"""
    tfrecords.create_tfrecords('/Users/ben/Downloads/hipergator/2019_SRER_3_521000_3521000_image.tif', patch_size=400, patch_overlap=0.05, savedir="output",overwrite=True)
    for i in np.arange(729):
        assert os.path.exists("output/2019_SRER_3_521000_3521000_image/2019_SRER_3_521000_3521000_image_{}.png".format(i))
        
