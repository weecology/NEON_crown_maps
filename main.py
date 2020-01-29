import glob
import os
from start_cluster import GPU_cluster
from distributed import wait

def run(records, raster_dir):
    from comet_ml import Experiment    
    comet_experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                              project_name="frenchguiana", workspace="bw4sz")

    from deepforest import deepforest
    import predict

    #Create model and set config
    model = deepforest.deepforest()
    model.use_release()
    print(model.config)
    model.config["batch_size"] = 25
    
    #Predict
    comet_experiment.log_parameters(model.config)
    predict.predict_tiles(model, [records], patch_size=400, raster_dir=[raster_dir], save_dir="/orange/ewhite/b.weinstein/NEON/predictions/", batch_size=model.config["batch_size"])

#f
if __name__ == "__main__":
    #RGB files
    rgb_list = glob.glob("/orange/ewhite/NeonData/**/*image.tif",recursive=True)
    
    #tfrecord files
    tfrecord_list = glob.glob("/orange/ewhite/b.weinstein/NEON/crops/*.tfrecord")
    
    #match rgb list to tfrecords
    rgb_names = [os.path.splitext(os.path.basename(x))[0] for x in rgb_list]
    tfrecord_names = [os.path.splitext(os.path.basename(x))[0] for x in tfrecord_list]
    
    indices = [rgb_names.index(x) for x in tfrecord_names]
    raster_dir = [rgb_list[x] for x in indices]
    raster_dir = [os.path.dirname(x) for x in raster_dir]
    
    records = tfrecord_list[:50]
    raster_dir =raster_dir[:50]
    
    client = GPU_cluster(gpus=5)
    
    results = client.map(run, records,raster_dir)
    wait(results)
    
    
    