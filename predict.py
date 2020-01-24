import os
import tensorflow as tf
import pandas as pd
import dask
import numpy as np
import geopandas
import rasterio
import shapely

from . import tfrecords
from deepforest import deepforest

def create_model():
    model = deepforest.deepforest()
    model.use_release()
    return model

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
def predict_tiles(model, records,patch_size=400, batch_size=1,raster_dir="./", score_threshold=0.05,max_detections=300,classes={0:"Tree"}):
    """Parallel loop through tile list and predict tree crowns
    Args:
        tile_list: a list of paths on disk of RGB tiles to predict
        client: a dask client with workers waiting for tasks (optional)
    """ 
    #for each tfrecord create a tensor and step, might be more efficient to create a full set of tensors and 
    results = [ ]    
    for tfrecord in records:
        result = predict_tile(model=model, tfrecord=tfrecord, patch_size=patch_size, raster_dir=raster_dir, batch_size=batch_size, score_threshold=score_threshold, max_detections=max_detections, classes=classes)
        results.append(result)
        
    full_df = pd.concat(results)
    return full_df

def predict_tile(model, tfrecord, patch_size, raster_dir, image_size=800, batch_size=1,score_threshold=0.05, max_detections=300, classes={0:"Tree"}):
    """Predict a tile of a tfrecords"""
    iterator = tfrecords.create_tensors(tfrecord, batch_size=batch_size)        
    record_results = [ ]        
    
    #predict tensor - throw error at end of record
    record_boxes = []
    record_scores = []
    record_labels = []
    
    #Iterate through tfrecord until the end
    while True:
        try:
            box_array, score_array, label_array = model.prediction_model.predict_on_batch(iterator)
            record_boxes.append(box_array)        
            record_scores.append(score_array)
            record_labels.append(label_array)
        
        except tf.errors.OutOfRangeError:
            break 
    
    #Number of batches produced, combine into one array
    n = len(record_boxes)
    
    #Process each prediction batch
    for index in np.arange(n):
        boxes  = record_boxes[index]
        scores  = record_scores[index]
        labels  = record_labels[index]
        
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
        
        #Change numberic class into string label
        df.label = df.label.astype(int)
        df.label = df.label.apply(lambda x: classes[x])
        df["filename"] = tfrecord
    
        record_results.append(df)

    #pandas frame
    record_df = pd.concat(record_results)
    mosaic_df = run_non_max_suppression(record_df)
    
    #Project results into UTM
    raster_name = "{}.tif".format(os.path.splitext(os.path.basename(tfrecord))[0])
    raster_path = "{}/{}".format(raster_dir,raster_name)
    mosaic_df = project(raster_path, mosaic_df)
    mosaic_df["filename"] = raster_name
    
    return mosaic_df
    
def project(raster_path, boxes):
    """Project boxes into utm"""
    with rasterio.open(raster_path) as dataset:
        bounds = dataset.bounds
        pixelSizeX, pixelSizeY  = dataset.res
    
    #subtract origin. Recall that numpy origin is top left! Not bottom left.
    boxes["xmin"] = (boxes["xmin"] *pixelSizeX) + bounds.left
    boxes["xmax"] = (boxes["xmax"] * pixelSizeX) + bounds.left
    boxes["ymin"] = bounds.top - (boxes["ymin"] * pixelSizeY) 
    boxes["ymax"] = bounds.top - (boxes["ymax"] * pixelSizeY)
    
    # combine column to a shapely Box() object, save shapefile
    boxes['geometry'] = boxes.apply(lambda x: shapely.geometry.box(x.xmin,x.ymin,x.xmax,x.ymax), axis=1)
    boxes = geopandas.GeoDataFrame(boxes, geometry='geometry')
    
    #set projection, (see dataset.crs) hard coded here
    boxes.crs = {'init' :"{}".format(dataset.crs)}
    
    return boxes
