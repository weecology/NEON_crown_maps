#LIDAR drape
'''
Drape shapefile predictions on to point cloud and extract height
'''
import geopandas as gp
import pandas as pd
from shapely import geometry
from matplotlib import pyplot
import re
import glob 
import numpy as np
import os
import cv2
import random
import rasterstats


r = lambda: random.randint(0,255)

def find_lidar_file(image_path, dirname):
    """
    Find the lidar file that matches RGB tile
    """
    #Look for lidar tile
    laz_files = glob.glob(dirname + "*.laz")
    
    #extract geoindex
    pattern = r"_(\d+_\d+_image.*)"
    match = re.findall(pattern, image_path)
    
    #replace .tif with .laz pattern
    match = [x.replace("image","classified_point_cloud") for x in match]
    match = [x.replace(".tif",".laz") for x in match]
    if len(match)>0:
        match = match[0]
    else:
        return None
    
    #Look for index in available laz
    laz_path = None
    for x in laz_files:
        if match in x:
            laz_path = x

    #Raise if no file found
    if not laz_path:
        print("No matching lidar file, check the lidar path: %s" %(dirname))
        FileNotFoundError
    
    return laz_path

def fetch_lidar_filename(row, dirname):
    """
    Find lidar path in a directory.
    param: row a dictionary with tile "key" for filename to be searched for
    return: string location on disk
    """
    
    #How to direct the lidar path to the right directory?
    direct_filename = os.path.join(dirname, os.path.splitext(row["tile"])[0] + ".laz")

    if os.path.exists(direct_filename):
        laz_path = direct_filename
    else:
        print("Filename: %s does not exist, searching within %s" %(direct_filename, dirname))        
        laz_path = find_lidar_file(image_path=row["tile"], dirname=dirname)
        
    return laz_path

def postprocess_CHM(shapefile, CHM, min_height):
    
    #Extract zonal stats
    boxes = gp.read_file(shapefile)    
    boxes[["left","bottom","right","top"]] = boxes[["left","bottom","right","top"]].astype(float)
    
    draped_boxes = rasterstats.zonal_stats(shapefile, CHM, stats="percentile_99")
    boxes["height"]  = [x["percentile_99"] for x in draped_boxes]
    
    #extract 
    #Rename column
    boxes = boxes[boxes.height > min_height]
    
    #Calculate crown area
    boxes["area"] = (boxes["top"] - boxes["bottom"]) * (boxes["right"] - boxes["left"])   
    
    return boxes

