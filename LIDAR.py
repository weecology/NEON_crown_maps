#LIDAR drape
'''
Drape shapefile predictions on to point cloud and extract height
'''
import geopandas as gp
import pandas as pd
import pyfor
from shapely import geometry
from matplotlib import pyplot
import re
import glob 
import numpy as np
import os
import cv2
import random

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

def load_lidar(laz_path, normalize=True):
    """
    Load lidar tile from file based on path name
    laz_path: A string of the file path of the lidar tile
    normalize: Perform ground normalization (slow)
    return: A pyfor point cloud
    """
    
    try:
        pc = pyfor.cloud.Cloud(laz_path)
        pc.extension = ".las"
        
    except FileNotFoundError:
        print("Failed loading path: %s" %(laz_path))
        return None
        
    #normalize and filter
    if normalize:
        try: 
            pc.normalize(0.33)
        except:
            print("No vertical objects in image, skipping normalization")
            #TODO use NEON classification for normalization
            return None
    
    #Quick filter for unreasonable points.
    pc.filter(min = -1, max = 100 , dim = "z")    
    
    #Check dim
    assert (not pc.data.points.shape[0] == 0), "Lidar tile is empty!"
    
    return pc
         
def check_density(pc, bounds=[]):
    ''''
    Check the point density of a pyfor point cloud
    bounds: a utm array [xmin, xmax, ymin, ymax] to limit density search
    returns: density in points/m^2
    '''
    if len(bounds) > 0:
        #Filter by utm bounds, find points in crop
        xmin, xmax, ymin, ymax = bounds
        filtered_points = pc.data.points[(pc.data.points.x > xmin) & (pc.data.points.x < xmax)  & (pc.data.points.y > ymin) & (pc.data.points.y < ymax)]
        n_points = filtered_points.shape[0]
        assert n_points > 0, "No points remain after bounds filter"
        
    else:
        #number of points
        n_points =  pc.data.points.shape[0]
        
        #area
        xmin = pc.data.x.min()
        xmax = pc.data.x.max()

        ymin = pc.data.y.min()
        ymax = pc.data.y.max()
    
    area = (xmax - xmin) * (ymax - ymin)
    
    density = n_points / area
    
    return density

def lookup_height(left, right, bottom, top,pc):
    """Find max return height for each predicted tree
    row: a geopandas dataframe row of bounding boxes
    pc: a pyfor point cloud
    """
    box_points  = pc.data.points.loc[(pc.data.points.x > left) &
                                         (pc.data.points.x < right)  &
                                         (pc.data.points.y >bottom)   &
                                         (pc.data.points.y < top)]

    max_height = box_points.z.max()     
    return max_height

def drape_boxes(boxes, pc, min_height=3):
    '''
    boxes: geopandas dataframe of predictions from DeepForest
    pc: Optional point cloud from memory, on the fly generation
    bounds: optional utm bounds to restrict utm box
    '''
    #Get max height of tree box    
    boxes["height"] = boxes.apply(lambda row: lookup_height(row["left"], row["right"],row["bottom"],row["top"], pc),axis=1)
        
    #remove boxes too small
    boxes = boxes[boxes["height"]>min_height]
    
    return boxes    
    
def postprocess(shapefile, pc, bounds=None):
    """
    Drape a shapefile of bounding box predictions over LiDAR cloud
    """
    #Read shapefile
    df = gp.read_file(shapefile)
    
    #Convert data types
    boxes[["left","bottom","right","top"]] = boxes[["left","bottom","right","top"]].astype(float)
    
    #Drape boxes
    boxes = drape_boxes(boxes=df, pc = pc, min_height=3)     
    
    #Calculate crown area
    boxes["area"] = (boxes["top"] - boxes["bottom"]) * (boxes["left"] - boxes["right"])
    
    return boxes
    