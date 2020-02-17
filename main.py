import glob
import os
from start_cluster import start
from distributed import wait, as_completed
import numpy as np
import random
import re

def lookup_rgb_path(tfrecord,rgb_list):
    #match rgb list to tfrecords
    rgb_names = [os.path.splitext(os.path.basename(x))[0] for x in rgb_list]
    tfrecord_name = os.path.splitext(os.path.basename(tfrecord))[0]
    
    #Find raster directory for each rgb file
    index = rgb_names.index(tfrecord_name)
    rgb_path = rgb_list[index]
    
    return rgb_path
    
def find_files(site_regex = None, target_list=None):
    #Find available tiles
    tile_list = glob.glob("/orange/ewhite/NeonData/**/*image.tif",recursive=True)
    
    if target_list: 
        selected_index = []
        for target in target_list:
            i = np.where([bool(re.search(target,x)) for x in tile_list])[0][0]
            selected_index.append(i)
        
    #select by sites 
    if site_regex:
        selected_index = np.where([bool(re.search(site_regex,x)) for x in tile_list])
        tile_list = tile_list[selected_index]
    
    return tile_list

def year_filter(tile_list, year=None):
    
    if year:
        year="{}_".format(year)
        #Find available tiles
        selected_index = np.where([bool(re.search(year,x)) for x in tile_list])[0]
        tile_list = [tile_list[x] for x in selected_index]
    
    return tile_list

def generate_tfrecord(tile_list, client, n=None,site_regex=None, year=None, target_list=None):
    """Create tfrecords
    tile_list: list of rgb tiles to generate tfrecord
    client: dask client
    year: year filter
    n: number of tiles to limit for testing
    site_regex: regular expression to search tile paths (e.g "OSBS|HARV")
    target_list: an optional list of files to run
    """
    from utils import tfrecords
            
    #Find site files
    tile_list = find_files()
    
    #Select year
    tile_list = year_filter(tile_list, year=year)
    
    if n:
        random.shuffle(tile_list)
        tile_list = tile_list[:n]
    
    print("Running {} tiles: \n {} ...".format(len(tile_list),tile_list[:3]))    
    
    written_records = client.map(tfrecords.create_tfrecords, tile_list, patch_size=400, savedir="/orange/ewhite/b.weinstein/NEON/crops/")
    
    return written_records
    
def run_rgb(records, raster_dir):
    from deepforest import deepforest
    import predict
    import LIDAR

    #Create model and set config
    model = deepforest.deepforest()
    model.use_release()
    model.config["batch_size"] = 32
    
    #Report config
    print(model.config)
    
    #Predict
    #comet_experiment.log_parameters(model.config)
    shp = predict.predict_tiles(model, [records], patch_size=400, raster_dir=[raster_dir], save_dir="/orange/ewhite/b.weinstein/NEON/predictions/", batch_size=model.config["batch_size"])
    
    return shp[0]

def run_lidar(shp,lidar_list, min_height =2, save_dir=""):
    """
    shp: path to a DeepForest prediction shapefile
    lidar_list: list of a lidar files to look up corresponding laz file
    min_height: minimum height of a tree to consider
    This function is written to function as results are completled, so the laz file cannot be anticipated
    """
    import LIDAR
    
    #Get geoindex
    lidar_name = [os.path.splitext(os.path.basename(x))[0] for x in lidar_list]
    geo_index = re.search("(\d+_\d+)_image",shp).group(1)
    index = np.where([geo_index in x for x in lidar_name])
    
    if len(index) > 1:
        raise ValueError("SHP file {} matches more than one .laz file".format(shp))
    else:
        #Tuple to numeric
        index = list(index)[0][0]
        
    # lookup Lidar path
    laz_path = lidar_list[index]
    
    #Load point cloud
    pc = LIDAR.load_lidar(laz_path)
    
    #Drape and collect height information
    boxes = LIDAR.postprocess(shp, pc, min_height=min_height)
    
    #Save shapefile
    bname = os.path.basename(shp)
    postprocessed_filename = os.path.join(save_dir, bname)
    boxes.to_file(postprocessed_filename, driver='ESRI Shapefile')
    
    return postprocessed_filename
    
if __name__ == "__main__":
    
    #Create dask clusters
    cpu_client = start(cpus = 20)
    
    gpu_client = start(gpus=4)
    
    #File lists
    rgb_list = glob.glob("/orange/ewhite/NeonData/**/*image.tif",recursive=True)
    lidar_list = glob.glob("/orange/ewhite/NeonData/**/ClassifiedPointCloud/*.laz",recursive=True)
    
    #Create tfrecords, either specify a set of tiles or sample random
    
    target_list =[
    "2019_WREF_3_582000_5073000_image.tif",
    "2018_ABBY_2_557000_5065000_image.tif",
    "2018_CLBJ_3_627000_3694000_image.tif",
    "2018_GRSM_4_273000_3954000_image.tif",
    "2018_OSBS_4_400000_3285000_image.tif",
    "2018_SOAP_3_303000_4099000_image.tif",
    "2018_SRER_2_503000_3520000_image.tif",
    "2019_DSNY_5_462000_3100000_image.tif",
    "2019_NOGP_3_353000_5187000_image.tif",
    "2019_SERC_4_364000_4308000_image.tif",
    "2019_TALL_5_465000_3646000_image.tif",
   "2019_TEAK_4_315000_4104000_image.tif"
    ]
    
    generated_records = generate_tfrecord(rgb_list, cpu_client,  n= 50, target_list = map_box, site_regex=None)
    
    predictions = []    
    
    #As records are created, predict.
    for future, result in as_completed(generated_records, with_results=True):
        
        print("Running prediction for completed future with tfrecord {}".format(result))
        #Lookup rgb path to create tfrecord. If it was a blank tile, result will be Nonetype
        if result:
            rgb_path = lookup_rgb_path(tfrecord = result, rgb_list = rgb_list)
        else:
            continue
        
        #Split into basename and dir
        rgb_name = os.path.splitext(os.path.basename(rgb_path))[0]
        raster_dir = os.path.dirname(rgb_path)
        
        #Predict record
        result = gpu_client.submit(run_rgb, result, raster_dir)
        predictions.append(result)
    
    #As predictions complete, run postprocess to drape LiDAR and extract height
    draped_files = [ ]
    for future, result in as_completed(predictions, with_results=True):
        try:
            print("Postprocessing: {}".format(result))                    
            postprocessed_filename = cpu_client.submit(run_lidar, result, lidar_list=lidar_list, save_dir="/orange/ewhite/b.weinstein/NEON/draped/")
        except:
            result.traceback()
        draped_files.append(postprocessed_filename)
    
    wait(draped_files)
    
    
    