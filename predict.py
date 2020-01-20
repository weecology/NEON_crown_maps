from deepforest import deepforest
import pandas as pd
from . import tfrecords
import dask
import numpy as np
import geopandas

def create_model():
    model = deepforest.deepforest()
    model.use_release()
    return model

#predict a tile
def predict_tile(record,patch_size=400, batch_size=1, score_threshold=0.05,max_detections=300,classes={"Tree":0}):
    """Parallel loop through tile list and predict tree crowns
    Args:
        tile_list: a list of paths on disk of RGB tiles to predict
        client: a dask client with workers waiting for tasks (optional)
    """
    #Load model
    model = create_model()
    
    #Create tfrecord tensor
    dataset_tensor = tfrecords.create_tensors(record, batch_size=batch_size)
    
    #predict tensor
    boxes, scores, labels = model.prediction_model.predict(dataset_tensor,steps=1)
    
    # correct boxes for image scale
    #TODO make dynamic
    scale = float(800/patch_size)
    boxes /= scale

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those scores
    scores = scores[0][indices]

    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores)[:max_detections]

    # select detections
    image_boxes      = boxes[0, indices[scores_sort], :]
    image_scores     = scores[scores_sort]
    image_labels     = labels[0, indices[scores_sort]]
    image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

    df = pd.DataFrame(image_detections, columns = ["xmin","ymin","xmax","ymax","score","label"])
    
    #Change numberic class into string label
    df.label = df.label.astype(int)
    df.label = df.label.apply(lambda x: classes[x])
    
    #pandas frame
    return df

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
    
    #get proj info see:https://gis.stackexchange.com/questions/204201/geopandas-to-file-saves-geodataframe-without-coordinate-system
    prj = 'PROJCS["WGS_1984_UTM_Zone_22N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-51],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["Meter",1]]'
    boxes.to_file('PrebuiltModel.shp', driver='ESRI Shapefile',crs_wkt=prj)    
    
    return boxes
