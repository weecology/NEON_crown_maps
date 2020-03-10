# Analysis module
import glob
import geopandas
import numpy as np
from geopandas.tools import sjoin
import rtree
import re
from matplotlib import pyplot

def match_years(geo_index, shps):
    """Match tree records across years based on heurestic criteria"""
    
    #Find matching files
    matched_shps = [x for x in shps if geo_index in x]
    
    #Load shapefiles
    shapefiles = [ ]
    for shp in matched_shps:
        #Load data and give it site and year and tile labels
        df = geopandas.read_file(shp)
        geo_index = re.search("(\d+_\d+)_image",shp).group(1)
        df["geo_index"] = geo_index
        df["Year"] = re.search("(\d+)_(\w+)_\d_\d+_\d+_image.shp",shp).group(1)
        df["Site"] = re.search("(\d+)_(\w+)_\d_\d+_\d+_image.shp",shp).group(2)
        shapefiles.append(df)
    
    #Join features and create a blank IoU column
    joined_boxes = sjoin(shapefiles[0],shapefiles[1])
    joined_boxes["IoU"] = None
    
    #Find trees with more than one match
    index_counts = joined_boxes.index.value_counts()    
    multiple_matches = index_counts[index_counts>=1]
    
    for target_index in multiple_matches.index:
        
        target_tree = shapefiles[0][shapefiles[0].index==target_index].geometry
        potential_index =  joined_boxes[joined_boxes.index == target_index].index_right
        potential_trees = shapefiles[1][shapefiles[1].index.isin(potential_index)]
        
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
    threshold_boxes = joined_boxes[joined_boxes.IoU > 0.5]
    
    #difference in height
    threshold_boxes["Height_difference"] = threshold_boxes["height_right"] - threshold_boxes["height_left"]
    
    #remove outliers
    lower, upper = threshold_boxes.Height_difference.quantile([0.01,0.99]).values
    threshold_boxes = threshold_boxes[(threshold_boxes.Height_difference > lower) & (threshold_boxes.Height_difference < upper)]
    threshold_boxes.Height_difference.hist(bins=20)
    
    #By Height
    pyplot.style.use('seaborn-whitegrid')    
    pyplot.scatter(threshold_boxes.height_left, threshold_boxes.height_right, alpha=0.5) 
    pyplot.xlabel("Height 2018 (m)")
    pyplot.ylabel("Height 2019 (m)")
    
    print("Mean height difference is {}".format(threshold_boxes.Height_difference.mean()))
    
    pyplot.savefig("tree_height.png")
    return threshold_boxes
        