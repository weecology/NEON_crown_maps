#Hard Mining of uncertain frames from DeepForest predictions
import os
import glob
import geopandas
import numpy as np
import pandas as pd
import rasterio
from rasterio import windows as rwindow
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
    
    left, bottom, right, top = raster.bounds
    #Create windows
    windows = preprocess.compute_windows(raster, patch_size=patch_size,patch_overlap=patch_overlap)
    
    #For each window, grab predictions and find average score
    #Score dict
    scores = {}
    data = {}
    crop_index = {}
    
    #Create spatial index
    spatial_index = shp.sindex
    
    for index,window in enumerate(windows):
        window_xmin, window_ymin, width, height = window.getRect()
        print(index)
        #transform    
        xmin = (window_xmin * cell_size) + left
        xmax = (window_xmin + width) * cell_size  + left
        ymax = top  - (window_ymin * cell_size)  
        ymin = top - (window_ymin + height) * cell_size
        
        #Spatial clip to window using spatial index for faster querying
        possible_matches_index = list(spatial_index.intersection([xmin,ymin,xmax,ymax]))
        possible_matches = shp.iloc[possible_matches_index]
        
        #quick pass, some edges overlap
        try:
            filtered_boxes = possible_matches.cx[xmin:xmax,ymin:ymax]
        except:
                continue
        scores[index] = pd.to_numeric(filtered_boxes.score).mean()
        data[index] = filtered_boxes
        crop_index[index] = window
    
    #Find highest 5th and lowest fifth quartile
    score_df = pd.Series(scores)
    lowest = score_df.min()
    lowest_index = score_df[score_df == lowest].index[0]
    worst_window = data[lowest_index]
    
    #save RGB windows to file
    rasterio_window = rwindow.Window.from_slices(rows=crop_index[lowest_index].indices()[0],cols=crop_index[lowest_index].indices()[1])    
    crop_filename = os.path.join(save_dir,"{}_{}.tif".format(image_name,lowest_index)) 
    rwindow.transform(rasterio_window,raster.window_transform(rasterio_window))
    
    out_profile = raster.profile.copy()
    with rasterio.open(crop_filename, 'w',dtype=out_profile["dtype"],
                        driver="GTiff",
                        count=3,
                        crs=raster.crs,
                        width=rasterio_window.width,
                        transform=raster.window_transform(rasterio_window),
                        height=rasterio_window.height) as dst:
        RP1_block = raster.read(window=rasterio_window, masked=True)  
        dst.write(RP1_block)
        
    #Format annotations frame
    shp_filename = os.path.join(save_dir,"{}_{}.shp".format(image_name, lowest_index))    
    worst_window.to_file(shp_filename, driver='ESRI Shapefile')
    
if __name__ == "__main__":
    
    from start_cluster import start_dask_cluster
    from distributed import wait
    
    #Find files
    rgb_list = glob.glob("/orange/ewhite/NeonData/**/*image.tif",recursive=True)
    
    #tfrecord files
    predictions = glob.glob("/orange/ewhite/b.weinstein/NEON/predictions/*.shp")
    
    #match rgb list to tfrecords
    rgb_names = [os.path.splitext(os.path.basename(x))[0] for x in rgb_list]
    predictions_names = [os.path.splitext(os.path.basename(x))[0] for x in predictions]
    
    indices = [rgb_names.index(x) for x in predictions_names]
    image_paths = [rgb_list[x] for x in indices]
    
    client = start_dask_cluster(number_of_workers=5)
    futures = client.map(run,image_paths,predictions,save_dir="/orange/ewhite/b.weinstein/NEON/mining/")
    wait(futures)
