import glob
import time
import os
import re
import gc
import numpy as np
import random
from distributed import wait, as_completed
import pandas as pd
from start_cluster import start
import LIDAR
from utils import verify

def lookup_CHM_path(path, lidar_list, shp=True):
    """Find CHM file based on the image filename
    shp: Whether the input path is a shapefile (True)
    """
    #Get geoindex from path and match it to inventory of CHM tiles
    lidar_name = [os.path.splitext(os.path.basename(x))[0] for x in lidar_list]
    geo_index = re.search("(\d+_\d+)_image",path).group(1)
    CHM_path = [lidar_list[index] for index, x in enumerate(lidar_name) if geo_index in x]
    
    #If there are records, check that there is the correct year
    if CHM_path:
        #Match years 
        if shp:
            basename = os.path.basename(path)
            year = re.search("^(\d+)_",basename).group(1)
        else:
            year = re.search("/(\d+\\/FullSite)",path).group(1)
        
        CHM_path = [x for x in CHM_path if year in x]
        
        #Sanity check for length 1
        if len(CHM_path) is not 1:
            return None
        return CHM_path[0]
    else:
        return None

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
    
    print("There are {} files found in dir".format(len(tile_list)))
    
    RGB_verification = client.map(verify.check_RGB, tile_list)
    rgb_verified = [x.result() for x in RGB_verification]
    
    #Remove None
    rgb_verified  = [x for x in rgb_verified if x]
    print("There are {} verified RGB tiles before checking LiDAR".format(len(rgb_verified)))
    
    CHM_verification = [ ]
    for x in rgb_verified:
        try:
            lidar_path = lookup_CHM_path(x, lidar_pool, shp=False)
            if lidar_path:  
                check = client.submit(verify.check_CHM,lidar_path)
            else:
                print("{} has no matching CHM".format(x))
                check = False
            CHM_verification.append(check)
        except Exception as e:                
            print("Path CHM {} lookup failed with {}".format(check,e))
    
    CHM_verified = client.gather(CHM_verification)
    print("There are {} verified CHM tiles before checking matches".format(len(CHM_verified)))
    
    final_rgb_list = [rgb_verified[index] for index, x in enumerate(CHM_verified) if x]
    print("There are {} RGB tiles with matching verified CHMs".format(len(final_rgb_list)))
    
    df = pd.DataFrame({"path":final_rgb_list})
    
    df["year"] = df.path.apply(lambda x: verify.get_year(x))
    df["site"] = df.path.apply(lambda x: verify.get_site(x))
    site_totals = df.groupby(["site","year"]).size()
    print(site_totals)

    written_records = client.map(tfrecords.create_tfrecords, final_rgb_list, patch_size=400, patch_overlap=0.05, savedir="/orange/ewhite/b.weinstein/NEON/crops/",overwrite=overwrite)
    
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
    #Start GPU Client
    cpu_client = start(cpus = 80, mem_size ="7GB")
    gpu_client = start(gpus=10,mem_size ="12GB")    
 
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
        print("Submitted prediction for tfrecord {}, future is {}".format(result, gpu_result))        
        predictions.append(gpu_result)
            
    ###As predictions complete, run postprocess to drape LiDAR and extract height
    draped_files = [ ]
    for future in as_completed(predictions):
        try:
            result = future.result()
                       
            #Look up corresponding CHM path
            CHM_path = lookup_CHM_path(result, lidar_list,shp=True)
            
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

