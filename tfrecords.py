"""
Module: tfrecords

Tfrecords creation and reader for improved performance across multi-gpu
There were a tradeoffs made in this repo. It would be natural to save the generated prepreprocessed image to tfrecord from the generator. This results in enormous (100x) files. 
"""
import tensorflow as tf
import os
import csv
import numpy as np
from PIL import Image
import cv2

from deepforest import preprocess
from keras_retinanet.utils import image as keras_retinanet_image

def create_tf_example(fname):
    
    #Save image information and metadata so that the tensors can be reshaped at runtime
    example = tf.train.Example(features=tf.train.Features(feature={                     
        'image/filename':  tf.train.Feature(bytes_list=tf.train.BytesList(value=[fname.encode('utf-8')])),    
    }))
    
    return example

def create_tfrecords(tile_path,patch_size=400, patch_overlap=0.15, savedir="."):
    """
    Write crops to file and write a tfrecord file to use for tf dataset API
    Args:
        tile_path: Path on disk to .tif
        size: Number of images per tfrecord
        savedir: dir path to save tfrecords files
    
    Returns:
        written_files: A list of path names of written tfrecords
    """
    #Load image    
    raster = Image.open(tile_path)
    numpy_image = np.array(raster)
    image_name = os.path.splitext(os.path.basename(tile_path))[0]
    
    #Create window crop index
    windows = preprocess.compute_windows(numpy_image, patch_size,patch_overlap)
    written_files = []
    
    #Tensorflow writer
    tfrecord_filename = os.path.join(savedir, "{}.tfrecord".format(image_name))    
    tfwriter = tf.io.TFRecordWriter(tfrecord_filename)
    
    counter = 0
    for index in windows:
        #crop image
        crop = numpy_image[index.indices()] 
    
        #Crop and preprocess, resize
        crop        = keras_retinanet_image.preprocess_image(crop)
        crop, scale = keras_retinanet_image.resize_image(crop)    
        filename = os.path.join(savedir,"{}_{}.jpg".format(image_name,counter))
        
        #Write crop to file
        cv2.imwrite(img=crop,filename=filename)
        counter +=1 
        
        #Write tfrecord
        tf_example = create_tf_example(filename)
        tfwriter.write(tf_example.SerializeToString())
    
    return tfrecord_filename
        
#Reading
def _parse_fn(example):
    
    #Define features
    features = {
        'image/filename': tf.io.FixedLenFeature([], tf.string)
                        }
    
    # Load one example and parse
    example = tf.io.parse_single_example(example, features)
    
    #Load image from file
    filename = tf.cast(example["image/filename"],tf.string)    
    loaded_image = tf.read_file(filename)
    loaded_image = tf.image.decode_image(loaded_image, 3)
    
    ##Reshape to known shape
    #image_rows = tf.cast(example['image/height'], tf.int32)
    #image_cols = tf.cast(example['image/width'], tf.int32)    
    #loaded_image = tf.reshape(loaded_image, tf.stack([image_rows, image_cols, 3]), name="cast_loaded_image")            
    
    return loaded_image

def create_dataset(filepath, batch_size=1):
    """
    Args:
        filepath: list of tfrecord files
        batch_size: number of images per batch
        
    Returns:
        dataset: a tensorflow dataset object for model training or prediction
    """
    
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)
    
    ## This dataset will go on forever
    dataset = dataset.repeat()
        
    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      
    ## Set the batchsize
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    
    #Collect a queue of data tensors
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ## Create an iterator
    iterator = dataset.make_one_shot_iterator()    
    
    return iterator

def create_tensors(list_of_tfrecords):
    """Create a wired tensor target from a list of tfrecords
    
    Args:
        list_of_tfrecords: a list of tfrecord on disk to turn into a tfdataset
        
    Returns:
        inputs: input tensors of images
        targets: target tensors of bounding boxes and classes
        """
    #Create tensorflow iterator
    iterator = create_dataset(list_of_tfrecords)
    next_element = iterator.get_next()
    
    #Split into inputs and targets 
    inputs = next_element[0]

    return inputs