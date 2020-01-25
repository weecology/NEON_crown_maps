import glob
import predict
import os

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

records = tfrecord_list[:1]

#Create model and set config
model = predict.create_model()
print(model.config)
#model.config["multi-gpu"] = 2

#Predict
predict.predict_tiles(model, records, patch_size=800, raster_dir=raster_dir, save_dir="/orange/ewhite/b.weinstein/NEON/predictions/", batch_size=6)
