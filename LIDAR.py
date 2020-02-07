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

def createPolygon(xmin, xmax, ymin, ymax):
    '''
    Convert a pandas row into a polygon bounding box
    ''' 
    p1 = geometry.Point(xmin,ymax)
    p2 = geometry.Point(xmax,ymax)
    p3 = geometry.Point(xmax,ymin)
    p4 = geometry.Point(xmin,ymin)
    
    pointList = [p1, p2, p3, p4, p1]
    
    poly = geometry.Polygon([[p.x, p.y] for p in pointList])
    
    return poly


def get_window_extent(annotations, row, windows, rgb_res):
    '''
    Get the geographic coordinates of the sliding window.
    Be careful that the ymin in the geographic data refers to the utm (bottom) and the ymin in the cartesian refers to origin (top). 
    '''
    #Select tile from annotations to get extent
    tile_annotations = annotations[annotations["rgb_path"]==row["tile"]]
    
    #Set tile extent to convert to UTMs, flipped origin from R to Python
    tile_xmin = tile_annotations.tile_xmin.unique()[0]
    tile_ymax = tile_annotations.tile_ymax.unique()[0]
    tile_ymin = tile_annotations.tile_ymin.unique()[0]
    
    #Get window cartesian coordinates
    x,y,w,h= windows[row["window"]].getRect()
    
    window_utm_xmin = x * rgb_res + tile_xmin
    window_utm_xmax = (x+w) * rgb_res + tile_xmin
    window_utm_ymax = tile_ymax - (y * rgb_res)
    window_utm_ymin= tile_ymax - ((y+h) * rgb_res)
    
    return(window_utm_xmin, window_utm_xmax, window_utm_ymin, window_utm_ymax)
         
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

def drape_boxes(boxes, pc, bounds=[]):
    '''
    boxes: predictions from retinanet
    pc: Optional point cloud from memory, on the fly generation
    bounds: optional utm bounds to restrict utm box
    '''
    
    #reset user_data column
    pc.data.points.user_data =  np.nan
        
    tree_counter = 1
    for box in boxes:

        #Find utm coordinates
        xmin, xmax, ymin, ymax = find_utm_coords(box=box, pc=pc, bounds=bounds)
        
        #Get max height of tree box
        box_points  = pc.data.points.loc[(pc.data.points.x > xmin) & (pc.data.points.x < xmax)  & (pc.data.points.y > ymin)   & (pc.data.points.y < ymax)]
        
        max_height = box_points.z.max() 
        
        
        #Skip if under 3 meters
        if max_height < 3:
            continue
        else:
            #Update points
            pc.data.points.loc[(pc.data.points.x > xmin) & (pc.data.points.x < xmax)  & (pc.data.points.y > ymin)   & (pc.data.points.y < ymax),"user_data"] = tree_counter
            
            #update counter
            tree_counter +=1 

    return pc    
    
def find_utm_coords(box, pc, rgb_res = 0.1, bounds = []):
    
    """
    Turn cartesian coordinates back to projected utm
    bounds: an optional offset for finding the position of a window within the data
    """
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]
    
    #add offset if needed
    if len(bounds) > 0:
        tile_xmin, _ , _ , tile_ymax = bounds   
    else:
        tile_xmin = pc.data.points.x.min()
        tile_ymax = pc.data.points.y.max()
        
    window_utm_xmin = xmin * rgb_res + tile_xmin
    window_utm_xmax = xmax * rgb_res + tile_xmin
    window_utm_ymin = tile_ymax - (ymax * rgb_res)
    window_utm_ymax= tile_ymax - (ymin* rgb_res)     
        
    return(window_utm_xmin, window_utm_xmax, window_utm_ymin, window_utm_ymax)

def cloud_to_box(pc, bounds=[]):
    ''''
    pc: a pyfor point cloud with labeled tree data in the 'user_data' column.
    Turn a point cloud with a "user_data" attribute into a numpy array of boxes
    '''
    tree_boxes = [ ]
    
    tree_ids = pc.data.points.user_data.dropna().unique()
    
    #Try to follow order of input boxes, start at Tree 1.
    tree_ids.sort()
    
    #For each tree, get the bounding box
    for tree_id in tree_ids:
        
        #Select points
        points = pc.data.points.loc[pc.data.points.user_data == tree_id,["x","y"]]
        
        #turn utm to cartesian, subtract min x and max y value, divide by cell size. Max y because numpy 0,0 origin is top left. utm N is top. 
        #FIND UTM coords here
        if len(bounds) > 0:
            tile_xmin, _ , _ , tile_ymax = bounds     
            points.x = points.x - tile_xmin
            points.y = tile_ymax - points.y 
        else:
            points.x = points.x - pc.data.points.x.min()
            points.y = pc.data.points.y.max() - points.y 
        
            points =  points/ 0.1
        
        s = gp.GeoSeries(map(geometry.Point, zip(points.x, points.y)))
        point_collection = geometry.MultiPoint(list(s))        
        point_bounds = point_collection.bounds
        
        #if no area, remove treeID, just a single lidar point.
        if point_bounds[0] == point_bounds[2]:
            continue
        
        tree_boxes.append(point_bounds)
        
    #pass as numpy array
    tree_boxes =np.array(tree_boxes)
    
    return tree_boxes
    
def cloud_to_polygons(pc):
    ''''
    Turn a point cloud with a "Tree" attribute into 2d polygons for calculating IoU
    returns a geopandas frame of convex hulls
    '''
        
    hulls = [ ]
    
    tree_ids = pc.data.points.user_data.dropna().unique()
    
    for treeid in tree_ids:
        
        points = pc.data.points.loc[pc.data.points.user_data == treeid,["x","y"]].values
        s = gp.GeoSeries(map(geometry.Point, zip(points[:,0], points[:,1])))
        point_collection = geometry.MultiPoint(list(s))
        convex_hull = point_collection.convex_hull
        hulls.append(convex_hull)
        
    hulldf = gp.GeoSeries(hulls)
    
    return hulldf

def postprocess(image_boxes, pc, bounds=None):
    pc = drape_boxes(boxes=image_boxes, pc = pc, bounds=bounds)     

    #Get new bounding boxes
    image_boxes = postprocessing.cloud_to_box(pc, bounds)    
    image_scores = image_scores[:image_boxes.shape[0]]
    image_labels = image_labels[:image_boxes.shape[0]] 
    
    