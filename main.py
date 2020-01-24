import glob
import predict

if __name__ =="__main__":
    records = glob.glob("/orange/ewhite/b.weinstein/NEON/crops/*.tfrecord")
    records = records[:1]
    model = predict.create_model()
    print(model.config)
    #model.config["multi-gpu"] = 2
    predictions = predict.predict_tiles(model, records, patch_size=800, save_dir="/orange/ewhite/b.weinstein/NEON/predictions/", batch_size=8)
    