# Analysis module
import glob
import geopandas
import numpy as np
from geopandas.tools import sjoin
import rtree
from matplotlib import pyplot

def match_years(geo_index, shps):
    """Match tree records across years based on heurestic criteria"""
    
    #Find matching files
    matched_shps = [x for x in shps if geo_index in x]
    
    #Load shapefiles
    shapefiles = [ ]
    for shp in matched_shps:
        shapefiles.append(geopandas.read_file(shp))
    
    #Join features and create a blank IoU column
    joined_boxes = sjoin(shapefiles[0],shapefiles[1])
    joined_boxes["IoU"] = 0
    
    index_counts = joined_boxes.index.value_counts()
    index_counts.hist()
    
    multiple_matches = index_counts[index_counts>1]
    
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
        
        
    #remove trees with less than 0.3 IoU
    joined_boxes = joined_boxes[joined_boxes.IoU > 0.4]
    
    #difference in height
    joined_boxes["Height_difference"] = joined_boxes["height_right"] - joined_boxes["height_left"]
    
    joined_boxes.Height_difference.hist()
    
    pyplot.savefig("tree_height.png")
        