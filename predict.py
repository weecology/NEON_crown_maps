import os
import tensorflow as tf
import pandas as pd
import dask
import numpy as np
import geopandas
import rasterio
import shapely
from PIL import Image
from utils import tfrecords

from deepforest import deepforest
from deepforest import preprocess

def run_non_max_suppression(predicted_boxes, iou_threshold=0.15):
    """Run non-max supression on a pandas dataframe of results"""
    with tf.Session() as sess:
        print("{} predictions in overlapping windows, applying non-max supression".format(predicted_boxes.shape[0]))
        
        #Gather current predictions
        boxes = predicted_boxes[["xmin","ymin","xmax","ymax"]].values
        scores = predicted_boxes.score.values
        labels = predicted_boxes.label.values
        max_output_size=predicted_boxes.shape[0]
        
        #non-max suppression
        non_max_idxs = tf.image.non_max_suppression(boxes, scores, max_output_size=max_output_size, iou_threshold=iou_threshold)
        new_boxes = tf.cast(tf.gather(boxes, non_max_idxs), tf.int32)
        new_scores = tf.gather(scores, non_max_idxs)
        new_labels =  tf.gather(labels, non_max_idxs)
        
        #run tensors
        new_boxes, new_scores, new_labels = sess.run([new_boxes, new_scores, new_labels])
        
        #Reform dataframe
        image_detections = np.concatenate([new_boxes, np.expand_dims(new_scores, axis=1), np.expand_dims(new_labels, axis=1)], axis=1)
        mosaic_df = pd.DataFrame(image_detections,columns=["xmin","ymin","xmax","ymax","score","label"])
        mosaic_df.label = mosaic_df.label.str.decode("utf-8")
        print("{} predictions kept after non-max suppression".format(mosaic_df.shape[0]))
        
        return mosaic_df
    
#predict a set of tiles
def predict_tiles(model, records, patch_size=400, batch_size=1, raster_dir =["."], score_threshold=0.05,max_detections=300,classes={0:"Tree"},save_dir="."):
    """Parallel loop through tile list and predict tree crowns
    raster_dir: a list of directories to search for RGB image
    """ 
    #for each tfrecord create a tensor and step, might be more efficient to create a full set of tensors and 
    results = [ ]    
    for index, tfrecord in enumerate(records):
        print("Running index {}, record {}".format(index, tfrecord))
        
        #Refind raster path
        raster_name = os.path.splitext(os.path.basename(tfrecord))[0]        
        raster_path = os.path.join(raster_dir[index], "{}.tif".format(raster_name))
        
        #Run predictions
        boxes = predict_tile(model=model, tfrecord=tfrecord, patch_size=patch_size, batch_size=batch_size, score_threshold=score_threshold, max_detections=max_detections, classes=classes)
        
        if boxes.empty:
            print("No predictions in record {}, skipping...".format(tfrecord))
            continue
        
        #Project to utm
        projected_boxes = project(raster_path, boxes)
        
        #Shapefile file path
        shp_path = os.path.join(save_dir,'{}.shp'.format(raster_name))
        
        #Write
        projected_boxes.to_file(shp_path, driver='ESRI Shapefile')   
        print("{} written".format(shp_path))
        results.append(shp_path)
        
    return results
        
def parse_prediction(boxes, scores, labels, image_size, patch_size, score_threshold):
    """Reduce tf prediction into pandas dataframe"""
    #for each record
    scale = float(image_size/patch_size)

    # correct boxes for image scale
    boxes /= scale

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those scores
    scores = scores[0][indices]

    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores)

    # select detections
    image_boxes      = boxes[0, indices[scores_sort], :]
    image_scores     = scores[scores_sort]
    image_labels     = labels[0, indices[scores_sort]]
    image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

    df = pd.DataFrame(image_detections, columns = ["xmin","ymin","xmax","ymax","score","label"])
    
    return df

def predict_tile(model, tfrecord, patch_size=400, patch_overlap=0.05, image_size=800, batch_size=1,score_threshold=0.05, max_detections=300, classes={0:"Tree"}):
    """Predict a tile of a tfrecords windows
        Args:
            model: a deepforest model object
            tfrecord: a tfrecord with filenames of crops to run
            patch_size: size of the crops of each window in px 
            image_size: keras-retinanet resizing
            batch_size: number of windows to predict at once
            score_threshold: min label score to include
            max_detections: max number of detections per window
            classes: dictionary to turn integers into string labels
            """
    iterator = tfrecords.create_tensors(tfrecord, batch_size=batch_size, patch_size=patch_size)        
    record_results = [ ]        
    
    #read window metadata
    metadata_filename = "{}.csv".format(os.path.splitext(tfrecord)[0])
    metadata = pd.read_csv(metadata_filename)

    #Create window crop index
    #predict tensor - throw error at end of record
    record_boxes = []
    record_scores = []
    record_labels = []

    #Iterate through tfrecord until the end
    counter=0
    while True:
        try:
            #Predict batch of data
            box_array, score_array, label_array = model.prediction_model.predict_on_batch(iterator)
            record_boxes.append(box_array)        
            record_scores.append(score_array)
            record_labels.append(label_array)
            counter+=1
        except tf.errors.OutOfRangeError: 
            print("{} batches in {}".format(counter,tfrecord))
            break
    
    #Number of batches produced, TODO combine into one array?
    record_boxes = np.concatenate(record_boxes)
    record_scores = np.concatenate(record_scores)
    record_labels = np.concatenate(record_labels)

    n = record_boxes.shape[0]
    
    #Parse predictions
    for index in np.arange(n):
        #Process each prediction into batch of 1        
        boxes  = record_boxes[index,:,:]
        boxes =  np.expand_dims(boxes, 0)
        
        scores  = record_scores[index,:]
        scores =  np.expand_dims(scores, 0)        
        
        labels  = record_labels[index,:]
        labels =  np.expand_dims(labels, 0)
        
        df = parse_prediction(boxes, scores, labels, image_size, patch_size, score_threshold)

        #Change numberic class into string label
        df.label = df.label.astype(int)
        df.label = df.label.apply(lambda x: classes[x])
        df["filename"] = tfrecord
        
        #Add to window extent, create original windows object (must be consistant with generate)
        #transform coordinates to original system
        window_df = metadata[metadata.window == index]
        xmin = window_df.xmin.values[0]
        ymin = window_df.ymin.values[0]
        df.xmin = df.xmin + xmin
        df.xmax = df.xmax + xmin
        df.ymin = df.ymin + ymin
        df.ymax = df.ymax + ymin
        
        record_results.append(df)

    #pandas frame
    record_df = pd.concat(record_results)
    mosaic_df = run_non_max_suppression(record_df)
    mosaic_df["filename"] = tfrecord    

    return mosaic_df
    
def project(raster_path, boxes):
    """Project boxes into utm"""
        
    with rasterio.open(raster_path) as dataset:
        bounds = dataset.bounds
        pixelSizeX, pixelSizeY  = dataset.res
    
    #subtract origin. Recall that numpy origin is top left! Not bottom left.
    boxes["left"] = (boxes["xmin"] *pixelSizeX) + bounds.left
    boxes["right"] = (boxes["xmax"] * pixelSizeX) + bounds.left
    boxes["top"] = bounds.top - (boxes["ymin"] * pixelSizeY) 
    boxes["bottom"] = bounds.top - (boxes["ymax"] * pixelSizeY)
    
    # combine column to a shapely Box() object, save shapefile
    boxes['geometry'] = boxes.apply(lambda x: shapely.geometry.box(x.left,x.top,x.right,x.bottom), axis=1)
    boxes = geopandas.GeoDataFrame(boxes, geometry='geometry')
    
    #set projection, (see dataset.crs) hard coded here
    boxes.crs = {'init' :"{}".format(dataset.crs)}
    
    #Select columns
    boxes = boxes[["left","bottom","right","top","score","label","geometry"]]    
    return boxes
