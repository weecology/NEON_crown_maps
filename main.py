import glob
import os
import re
import gc
from start_cluster import start
from distributed import wait, as_completed
import numpy as np
import random
import LIDAR
import time


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

def lookup_rgb_path(tfrecord,rgb_list):
    #match rgb list to tfrecords
    rgb_names = [os.path.splitext(os.path.basename(x))[0] for x in rgb_list]
    tfrecord_name = os.path.splitext(os.path.basename(tfrecord))[0]
    
    #Find raster directory for each rgb file
    index = rgb_names.index(tfrecord_name)
    rgb_path = rgb_list[index]
    
    return rgb_path
    
def find_files(tile_list, site_list = None, target_list=None, year_list=None):
    """Filter tiles to run based on filtered paths"""
    #Find available tiles
    
    if target_list: 
        tile_list = [x for x in tile_list for y in target_list if y in x]
        
    #select by sites 
    if site_list:
        tile_list = [x for x in tile_list for y in site_list if y in x]
    
    #select by years
    if year_list:
        year_list = ["{}_".format(x) for x in year_list]
        tile_list = [x for x in tile_list for y in year_list if y in x]        
        
    return tile_list

def generate_tfrecord(tile_list, client, n=None,site_list=None, year_list=None, target_list=None):
    """Create tfrecords
    tile_list: list of rgb tiles to generate tfrecord
    client: dask client
    year: year filter
    n: number of tiles to limit for testing
    site_list: list to search tile paths (e.g ["OSBS","HARV"])
    year_list: list to search tile paths (e.g. ["2019","2018"])
    target_list: an optional list of files to run, just the relative path names
    """
    from utils import tfrecords
            
    #Find site files
    tile_list = find_files(tile_list, site_list=site_list, year_list=year_list, target_list=target_list)
        
    if n:
        random.shuffle(tile_list)
        tile_list = tile_list[:n]
    
    print("Running {} tiles: \n {} ...".format(len(tile_list),tile_list[:10]))    
    
    written_records = client.map(tfrecords.create_tfrecords, tile_list, patch_size=400, patch_overlap=0.05, savedir="/orange/ewhite/b.weinstein/NEON/crops/")
    
    return written_records

def run_rgb(records, rgb_paths):
    from deepforest import deepforest
    from keras import backend as K            
    import predict

    #Create model and set config
    model = deepforest.deepforest()
    model.use_release()
    
    #A 1km tile has 729 windows, evenly divisible batches is 9 * 81 = 729
    model.config["batch_size"] = 3    
    
    #Predict
    shp = predict.predict_tiles(model, [records], patch_size=400, rgb_paths=[rgb_paths], save_dir="/orange/ewhite/b.weinstein/NEON/predictions/", batch_size=model.config["batch_size"])
    
    gc.collect()
    K.clear_session()
        
    return shp[0]

def run_lidar(shp, CHM_path, min_height=3, save_dir=""):
    """
    shp: path to a DeepForest prediction shapefile
    CHM_Path: path to the .tif height model
    min_height: minimum height of a tree to consider
    This function is written to function as results are completled, so the laz file cannot be anticipated
    """
    
    #Drape and collect height information
    boxes = LIDAR.postprocess_CHM(shp, CHM_path, min_height=min_height)
    
    #Save shapefile
    bname = os.path.basename(shp)
    postprocessed_filename = os.path.join(save_dir, bname)
    boxes.to_file(postprocessed_filename, driver='ESRI Shapefile')
    
    return postprocessed_filename
    
if __name__ == "__main__":
    
    #Create dask clusters
    cpu_client = start(cpus = 15, mem_size ="11GB")
    
    gpu_client = start(gpus=5,mem_size ="15GB")
    
    #File lists
    rgb_list = glob.glob("/orange/ewhite/NeonData/**/Mosaic/*image.tif",recursive=True)
    lidar_list = glob.glob("/orange/ewhite/NeonData/**/CanopyHeightModelGtif/*.tif",recursive=True)
    
    #Create tfrecords, either specify a set of tiles or sample random
    
    #target_list =[
    #"2019_WREF_3_582000_5073000_image.tif",
    #"2018_ABBY_2_557000_5065000_image.tif",
    #"2018_CLBJ_3_627000_3694000_image.tif",
    #"2018_OSBS_4_400000_3285000_image.tif",
    #"2018_SRER_2_503000_3520000_image.tif",
    #"2019_DSNY_5_462000_3100000_image.tif",
    #"2019_NOGP_3_353000_5187000_image.tif",
    #"2019_SERC_4_364000_4308000_image.tif",
    #"2019_TALL_5_465000_3646000_image.tif",
   #"2019_TEAK_4_315000_4104000_image.tif",
   #"2019_KONZ_5_704000_4335000_image.tif",
   #"2018_BART_4_317000_4874000_image.tif",
   #"2019_DELA_5_421000_3606000_image.tif",
    #"2019_BONA_3_476000_7233000_image.tif"]
    
    target_list = None
    site_list = ["BART","TEAK"]
    year_list = ["2019","2018","2017"]
    generated_records = generate_tfrecord(rgb_list, cpu_client, n=10, target_list = target_list, site_list=site_list, year_list=year_list)
    
    predictions = []    
    
    #As records are created, predict.
    for future, result in as_completed(generated_records, with_results=True):
        
        #Lookup rgb path to create tfrecord. If it was a blank tile, result will be Nonetype
        if result:
            rgb_path = lookup_rgb_path(tfrecord = result, rgb_list = rgb_list)
        else:
            print("future {} had no tfrecord generated".format(future))
            continue
                
        #Predict record
        gpu_result = gpu_client.submit(run_rgb, result, rgb_path)
        print("Completed prediction for tfrecord {}, future index is {}".format(result, gpu_result))        
        predictions.append(gpu_result)
            
    ###As predictions complete, run postprocess to drape LiDAR and extract height
    draped_files = [ ]
    for future in as_completed(predictions):
        try:
            result = future.result()
            CHM_path = lookup_CHM_path(result, lidar_list)
            postprocessed_filename = cpu_client.submit(run_lidar, result, CHM_path=CHM_path, save_dir="/orange/ewhite/b.weinstein/NEON/draped/")
            print("Postprocessing complete: {}".format(postprocessed_filename))                           
            draped_files.append(postprocessed_filename)            
        except Exception as e:
            print("Lidar draping future: {} failed with {}".format(future, e))   
            e.with_traceback(result.traceback)
    
    wait(draped_files)
    #Give it some time to cleanup
    time.sleep(20)
