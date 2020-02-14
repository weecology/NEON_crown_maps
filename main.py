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
    rgb_path = rgb_list[x]
    
    return rgb_path
    
def find_files(site_regex = None):
    #Find available tiles
    tile_list = glob.glob("/orange/ewhite/NeonData/**/*image.tif",recursive=True)
    
    #select by sites 
    if site_regex:
        selected_index = np.where([bool(re.search(site_regex,x)) for x in tile_list])
        tile_list = tile_list[selected_index]
    
    return tile_list

def generate_tfrecord(tile_list, client, n=None,site_regex=None):
    """Create tfrecords
    tile_list: list of rgb tiles to generate tfrecord
    client: dask client
    n: number of tiles to limit for testing
    site_regex: regular expression to search tile paths (e.g "OSBS|HARV")
    """
    
    from deepforest import tfrecords
    
    #Find files
    tile_list = find_files(site_regex)
    
    if n:
        random.shuffle(tile_list)
        tile_list = tile_list[:n]
    
    print("Running {} tiles: \n {}".format(len(tile_list),tile_list))    
    
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
    
    return shp

def run_lidar(shp,lidar_list, save_dir=""):
    """
    shp: path to a DeepForest prediction shapefile
    lidar_list: list of a lidar files to look up corresponding laz file
    This function is written to function as results are completled, so the laz file cannot be anticipated
    """
    import LIDAR
    
    #Get geoindex
    lidar_name = os.path.splitext(os.path.basename(lidar_list))[0] 
    geo_index = re.search("(\d+_\d+)_image",lidar_name).group(1)
    index = np.where([geo_index in x for x in lidar_name])
        
    # lookup Lidar path
    laz_path = lidar_list[index]
    
    #Load point cloud
    pc = LIDAR.load_lidar(laz_path)
    
    #Drape and collect height information
    boxes = LIDAR.postprocess(shp, pc)
    
    #Save shapefile
    bname = os.path.basename(shp)
    postprocessed_filename = os.path.join(save_dir, bname)
    boxes.to_file(postprocessed_filename, driver='ESRI Shapefile')
    
    return postprocessed_filename
    
if __name__ == "__main__":
    
    #Create dask clusters
    cpu_client = start(cpus = 10)
    
    gpu_client = start(gpus=3)
    
    #File lists
    rgb_list = glob.glob("/orange/ewhite/NeonData/**/*image.tif",recursive=True)
    lidar_list = glob.glob("/orange/ewhite/NeonData/**/ClassifiedPointCloud/*.laz",recursive=True)
    
    #Create tfrecords
    generated_records = generate_tfrecord(rgb_list, cpu_client, site_regex=None, n= 50)
    
    predictions = []    
    
    #As records are created, predict.
    for future, result in as_completed(generated_records, with_results=True):
        
        #Lookup rgb path to create tfrecord
        rgb_path = lookup_rgb_path(tfrecord = result, rgb_list = rgb_list)
        
        #Split into basename and dir
        rgb_name = os.path.splitext(os.path.basename(rgb_path))[0]
        raster_dir = os.path.dirname(rgb_path)
        
        #Predict record
        result = gpu_client.submit(run_rgb, result, raster_dir)
        predictions.append(result)
    
    #As predictions complete, run postprocess to drape LiDAR and extract height
    for future, result in as_completed(results, with_results=True):
        postprocessed_filename = cpu_client.submit(run_lidar, result, lidar_list, save_dir="/orange/ewhite/b.weinstein/NEON/draped/")
        print("Postprocessed: {}".format(postprocessed_filename))
        
    #Wait until all futures are complete
    wait(results)
    
    
    