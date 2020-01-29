from comet_ml import Experiment
import glob
import predict
import os
from deepforest import deepforest
import start_cluster
import numpy as np

#Variables
GPUS = 2

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

records = tfrecord_list[:10]

comet_experiment.log_parameter("Number of tiles",len(records))

#Splits into chunks of size GPU
record_list = list(zip(*[iter(records)]*int(len(records)/GPUS)))

#Start GPU cluster
client = GPU_cluster(gpus=2)

#Create model to get path to use release
def run(records):
    comet_experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                  project_name="frenchguiana", workspace="bw4sz")
    
    model = deepforest.deepforest()
    model.use_release()
    
    print(model.config)
    model.config["batch_size"] = 2
    
    #Predict
    predict.predict_tiles(model, records, patch_size=400, raster_dir=raster_dir, save_dir="/orange/ewhite/b.weinstein/NEON/predictions/", batch_size=model.config["batch_size"])

client.map(run, record_list)