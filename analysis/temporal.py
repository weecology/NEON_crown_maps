# Analysis module
import geopandas
import numpy as np
import re
import os
import rasterstats
from geopandas.tools import sjoin

def lookup_CHM_path(shp_path, lidar_list):
    """Find CHM file based on the shp filename"""
    
    #Get geoindex from shapefile and match it to inventory of CHM rifles
    lidar_name = [os.path.splitext(os.path.basename(x))[0] for x in lidar_list]
    geo_index = re.search("(\d+_\d+)_image",shp_path).group(1)
    index = np.where([geo_index in x for x in lidar_name])
    
    if len(index) == 0:
        raise ValueError("SHP file {} has no CHM matching file".format(shp_path))
    if len(index) > 1:
        raise ValueError("SHP file {} matches more than one .tif CHM file".format(shp_path))
    else:
        #Tuple to numeric
        index = list(index)[0][0]
        
    # lookup Lidar CHM path
    CHM_path = lidar_list[index]
    
    return CHM_path

def match_years(geo_index, shps, savedir = "."):
    """Match tree records across years based on heurestic criteria"""
    
    #Find matching files
    matched_shps = [x for x in shps if geo_index in x]
    
    #Load shapefiles
    shapefiles = { }
    for shp in matched_shps:
        #Load data and give it site and year and tile labels
        df = geopandas.read_file(shp)
        geo_index = re.search("(\d+_\d+)_image",shp).group(1)
        df["shp_path"] = shp
        df["geo_index"] = geo_index
        df["Year"] = re.search("(\d+)_(\w+)_\d_\d+_\d+_image.shp",shp).group(1)
        df["Site"] = re.search("(\d+)_(\w+)_\d_\d+_\d+_image.shp",shp).group(2)
        shapefiles[df["Year"].unique()[0]] = df   
    
    if not all([x in ["2018","2019"] for x in shapefiles.keys()]):
        raise ValueError("{} does not have data from 2018 and 2019: {}".format(geo_index,shapefiles.keys()))
    
    #Join features and create a blank IoU column
    joined_boxes = sjoin(shapefiles["2018"],shapefiles["2019"])
    joined_boxes["IoU"] = None
    
    #Find trees with more than one match
    index_counts = joined_boxes.index.value_counts()    
    multiple_matches = index_counts[index_counts>=1]
    
    for target_index in multiple_matches.index:
        
        target_tree = shapefiles["2018"][shapefiles["2018"].index==target_index].geometry
        potential_index =  joined_boxes[joined_boxes.index == target_index].index_right
        potential_trees = shapefiles["2019"][shapefiles["2019"].index.isin(potential_index)]
        
        #Calc IoU for each overlapped tree
        iou_list = []
        for _, row in potential_trees.iterrows():
            test_poly = row.geometry
            intersection = target_tree.intersection(test_poly).area
            union = target_tree.union(test_poly).area
            iou = intersection/float(union)
            iou_list.append(iou.values[0])
            
        #get highest overlap
        max_overlap_index = np.argmax(np.array(iou_list))
        matched_index = potential_trees.index[max_overlap_index]
        
        #Update value
        joined_boxes.loc[(joined_boxes.index == target_index) & (joined_boxes.index_right == matched_index),"IoU"] = iou_list[max_overlap_index]
        
    #remove trees with less than threshold IoU
    threshold_boxes = joined_boxes[joined_boxes.IoU > 0.4]
        
    #difference in height
    threshold_boxes["Height_difference"] = threshold_boxes["height_right"] - threshold_boxes["height_left"]
    
    #remove outliers
    lower, upper = threshold_boxes.Height_difference.quantile([0.01,0.99]).values
    threshold_boxes = threshold_boxes[(threshold_boxes.Height_difference > lower) & (threshold_boxes.Height_difference < upper)]
    
    fname = os.path.basename(shapefiles["2019"]["shp_path"].unique()[0])
    fname = os.path.splitext(fname)[0]
    fname = "{}/{}_growth.shp".format(savedir,fname)
    threshold_boxes.to_file(fname)
    
    return fname
        
def tree_falls(geo_index, shps, CHMs,savedir="."):
    """Find predictions that don't match accross years, where there is significant height drop among years
    geo_index: NEON geoindex to process
    shps: List of shapefiles to search
    CHMs: List of canopy height models to search
    """
    #Find matching shapefiles
    matched_shps = [x for x in shps if geo_index in x]
    
    #Load shapefiles
    shapefiles = {}
    for shp in matched_shps:
        #Load data and give it site and year and tile labels
        df = geopandas.read_file(shp)
        geo_index = re.search("(\d+_\d+)_image",shp).group(1)
        df["shp_path"] = shp
        df["geo_index"] = geo_index
        df["Year"] = re.search("(\d+)_(\w+)_\d_\d+_\d+_image.shp",shp).group(1)
        df["Site"] = re.search("(\d+)_(\w+)_\d_\d+_\d+_image.shp",shp).group(2)
        shapefiles[df["Year"].unique()[0]] = df   
    
    #Join to find predictions that don't match
    joined_boxes = sjoin(shapefiles["2018"],shapefiles["2019"])
    no_matches = shapefiles["2018"][~(shapefiles["2018"].index.isin(joined_boxes.index))]
    
    #For each tree that does not match, check the 2019 height
    CHM = lookup_CHM_path(shapefiles["2018"]["shp_path"].unique()[0], CHMs)
    
    if not os.path.exists(CHM):
        raise IOError("{} does not exist".format(CHM))
    
    draped_2019 = rasterstats.zonal_stats(no_matches, CHM, stats="mean")
    no_matches["2019_height"]  = [x["mean"] for x in draped_2019]
    
    #Keep predictions whose mean height dropped by more than 50%
    no_matches["height_frac"] =  (no_matches["2019_height"] - no_matches["height"]) / no_matches["height"]
    fall_df = no_matches[no_matches["height_frac"] < -0.5]
    
    #Keep predictions whose original height was greater than 5m
    #fall_df = fall_df[fall_df.height > 5]
    
    #Write tree fall shapefile
    fname = os.path.basename(shapefiles["2019"]["shp_path"].unique()[0])
    fname = os.path.splitext(fname)[0]
    fname = "{}/{}_treefall.shp".format(savedir,fname)
    fall_df.to_file(fname)
    
    #get predictions whose height did not drop by more than 50%, indiciating poor matching
    non_fall_df = no_matches[~(no_matches["height_frac"] < -0.5)]
    
    #Keep predictions whose original height was greater than 5m
    #fall_df = fall_df[fall_df.height > 5]
    
    #Write tree fall shapefile
    fname = os.path.basename(shapefiles["2019"]["shp_path"].unique()[0])
    fname = os.path.splitext(fname)[0]
    fname = "{}/{}_incorrect_treefall.shp".format(savedir,fname)
    non_fall_df.to_file(fname)
    
    return fname

