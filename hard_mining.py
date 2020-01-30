#Hard Mining
import os
import geopandas
import rasterio
import cv2
import pandas as pd

from deepforest import preprocess

def run(image_path,prediction_path, save_dir=".", patch_size=400,patch_overlap=0.15):
    """Run Hard Negative Mining. Loop through a series of DeepForest predictions, find the highest and lowest scoring quantiles and save them for retraining
    """
    #Read shapefile
    shp = geopandas.read_file(prediction_path)
    
    #Read tif
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    raster = rasterio.open(image_path)
    
    cell_size = raster.res[0]
    
    utm_xmin, utm_ymin, utm_xmax, utm_ymax = raster.bounds
    #Create windows
    windows = preprocess.compute_windows(raster, patch_size=patch_size,patch_overlap=patch_overlap)
    
    #For each window, grab predictions and find average score
    #Score dict
    scores = {}
    data = {}
    for index,window in enumerate(windows):
        window_xmin, window_ymin, width, height = window.getRect()
        
        #transform    
        xmin = (window_xmin * cell_size) + utm_xmin
        xmax = (window_xmin + width) * cell_size  + utm_xmin
        ymin = (window_ymin) * cell_size  + utm_ymin
        ymax = (window_ymin + height) * cell_size  + utm_ymin
        
        #Spatial clip to window
        filtered_boxes = shp.cx[xmin:xmax, ymin:ymax]
        scores[index] = pd.to_numeric(filtered_boxes.score).mean()
        data[index] = filtered_boxes
    
    #Find highest 5th and lowest fifth quartile
    score_df = pd.DataFrame(scores)
    lowest = score_df.quantile(q=0.05)
    worst_windows = data[lowest.index]
    
    highest = score_df.quantile(q=0.95)
    best_windows = data[highest.index]
    
    #save RGB windows to file
    for index in highest.index:
        crop = raster[windows[index].indices()]
        crop_filename = os.path.join(save_dir,"{}.tif".format(image_name))
        cv2.imwrite(crop_filename, crop)
        
    #Format annotations frame