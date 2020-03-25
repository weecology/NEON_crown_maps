import glob
import os
import re
import gc
from start_cluster import start
from distributed import wait, as_completed
import numpy as np
import random
import LIDAR
from utils import verify
import time

def lookup_CHM_path(path, lidar_list):
    """Find CHM file based on the image filename"""
    
    #Get geoindex from path and match it to inventory of CHM rifles
    lidar_name = [os.path.splitext(os.path.basename(x))[0] for x in lidar_list]
    geo_index = re.search("(\d+_\d+)_image",path).group(1)
    CHM_path = [ lidar_list[index] for index, x in enumerate(lidar_name) if geo_index in x]
    
    #Match years        
    year = re.search("DP3.30010.001/(\d+\\/FullSite)",path).group(1)
    CHM_path = [x for x in CHM_path if year in x]
    
    if len(CHM_path) > 1:
        raise ValueError("CHM path has length > 1: {}".format(CHM_path))
    
    return CHM_path[0]

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

def generate_tfrecord(tile_list, lidar_pool, client, n=None,site_list=None, year_list=None, target_list=None, overwrite=True):
    """Create tfrecords
    tile_list: list of rgb tiles to generate tfrecord
    lidar_pool: a list of CHM files to search for corresponding record
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
    
    #Verify RGB records
    RGB_verification = client.map(verify.check_RGB, tile_list)
    rgb_list_verified = [x.result() for x in RGB_verification]
    rgb_list_verified = [i for i in rgb_list_verified if i] 
    
    #Find corresponding CHM records
    futures = [ ]
    for path in rgb_list_verified:
        print(path)
        lidar_path = lookup_CHM_path(path, lidar_pool)
        print(lidar_path)
        future = client.submit(verify.check_CHM, lidar_path)
        futures.append(future)
    
    chm_verfied = client.gather(futures)
    
    #Filter out RGB tiles that have no CHM    
    rgb_list_verified = [rgb_list_verified[index] for index, x in enumerate(chm_verfied) if not x==None]
    
    print("Running {} verified tiles: \n {} ...".format(len(rgb_list_verified),rgb_list_verified[:10]))    
    
    written_records = client.map(tfrecords.create_tfrecords, rgb_list_verified, patch_size=400, patch_overlap=0.05, savedir="/orange/ewhite/b.weinstein/NEON/crops/",overwrite=overwrite)
    
    return written_records

def run_rgb(records, rgb_paths, overwrite=True, save_dir = "/orange/ewhite/b.weinstein/NEON/predictions/"):
    from deepforest import deepforest
    from keras import backend as K            
    import predict

    #Create model and set config
    model = deepforest.deepforest(weights= '/home/b.weinstein/miniconda3/envs/crowns/lib/python3.7/site-packages/deepforest/data/NEON.h5')
    
    #A 1km tile has 729 windows, evenly divisible batches is 9 * 81 = 729
    model.config["batch_size"] = 9    
    
    #Predict
    shp = predict.predict_tiles(model, [records], patch_size=400, rgb_paths=[rgb_paths], save_dir=save_dir, batch_size=model.config["batch_size"],overwrite=overwrite)
    
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
    cpu_client = start(cpus = 3, mem_size ="10GB")
    gpu_client = start(gpus=1,mem_size ="12GB")
 
    #Overwrite existing file?
    overwrite=True
    
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
    site_list = ["OSBS","DELA","BART","TEAK","BONA","SOAP","WREF"]
    year_list = ["2019","2018"]
    generated_records = generate_tfrecord(tile_list=rgb_list,
                                          lidar_pool=lidar_list,
                                          client=cpu_client,
                                          n=10,
                                          target_list = target_list,
                                          site_list=site_list,
                                          year_list=year_list,
                                          overwrite=overwrite)
    
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
        gpu_result = gpu_client.submit(run_rgb, result, rgb_path,overwrite=overwrite)
        print("Completed prediction for tfrecord {}, future index is {}".format(result, gpu_result))        
        predictions.append(gpu_result)
            
    ###As predictions complete, run postprocess to drape LiDAR and extract height
    draped_files = [ ]
    for future in as_completed(predictions):
        try:
            result = future.result()
            
            #Look up corresponding CHM path
            CHM_path = lookup_CHM_path(result, lidar_list)
            
            if not CHM_path:
                raise IOError("Image file: {} has no matching CHM".format(result))
            
            #Submit draping future
            postprocessed_filename = cpu_client.submit(run_lidar, result, CHM_path=CHM_path, save_dir="/orange/ewhite/b.weinstein/NEON/draped/")
            
            print("Postprocessing submitted: {}".format(result))                           
            draped_files.append(postprocessed_filename)            
        except Exception as e:
            print("Lidar draping future: {} failed with {}".format(future, e.with_traceback(future.traceback())))   
    
    wait(draped_files)
    
    #Give the scheduler some time to cleanup
    time.sleep(3)

