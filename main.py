from comet_ml import Experiment
import glob
import predict
import os

comet_experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                              project_name="frenchguiana", workspace="bw4sz")
 
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

comet_experiment.log_parameter("Number of tiles",len(tfrecord_list))
records = tfrecord_list[:10]

#Create model and set config
model = predict.create_model()
print(model.config)
model.config["multi-gpu"] = 1
model.config["batch_size"] = 25

#Predict
comet_experiment.log_parameters(model.config)
predict.predict_tiles(model, records, patch_size=400, raster_dir=raster_dir, save_dir="/orange/ewhite/b.weinstein/NEON/predictions/", batch_size=model.config["batch_size"])
