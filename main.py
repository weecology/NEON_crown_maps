import glob
import os
from start_cluster import GPU_cluster, start_dask_cluster
from distributed import wait, as_completed
import numpy as np

def run(records, raster_dir):
    #from comet_ml import Experiment    
    #comet_experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
    #                         project_name="frenchguiana", workspace="bw4sz")

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
    postprocessed_filename = os.path.join(save_dir, )
    boxes.to_file(postprocessed_filename, driver='ESRI Shapefile')
    
    return postprocessed_filename
    
if __name__ == "__main__":
    #RGB files
    rgb_list = glob.glob("/orange/ewhite/NeonData/**/*image.tif",recursive=True)
    
    #tfrecord files
    tfrecord_list = glob.glob("/orange/ewhite/b.weinstein/NEON/crops/*.tfrecord")
    
    #LiDAR files
    lidar_list = glob.glob("/orange/ewhite/NeonData/**/ClassifiedPointCloud/*.laz",recursive=True)
    
    #match rgb list to tfrecords
    rgb_names = [os.path.splitext(os.path.basename(x))[0] for x in rgb_list]
    tfrecord_names = [os.path.splitext(os.path.basename(x))[0] for x in tfrecord_list]
    
    #Find raster directory for each rgb file
    indices = [rgb_names.index(x) for x in tfrecord_names]
    raster_dir = [rgb_list[x] for x in indices]
    raster_dir = [os.path.dirname(x) for x in raster_dir]
    
    #Find LiDAR directory for each rgb file
    records = tfrecord_list[:50]
    raster_dir =raster_dir[:50]
    
    client = GPU_cluster(gpus=5, cpus = 10)
    
    results = client.map(run, records,raster_dir)
    
    for future, result in as_completed(results, with_results=True):
        postprocessed_filename = client.submit(lidar, result, lidar_list)
        print("Postprocessed: {}".format(postprocessed_filename))
    
    
    