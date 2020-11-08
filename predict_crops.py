import pandas as pd
import glob
from crown_maps import start_cluster

client = start_cluster.start(gpus=5)
client.wait_for_workers(2)

def load_model():
    from deepforest import deepforest    
    saved_model="/home/b.weinstein/miniconda3/envs/DeepTreeAttention_DeepForest/lib/python3.7/site-packages/deepforest/data/NEON.h5"
    model = deepforest.deepforest(saved_model = saved_model)
    
    return model

def run(x, model):
    boxes = model.predict_image(x, return_plot = False)
    boxes["file"] = x
    csv_name = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/pretraining/predictions/{}.csv".format(x)
    boxes.to_csv(csv_name)
    return x

files = glob.glob("/orange/ewhite/b.weinstein/NeonTreeEvaluation/pretraining/crops/*.jpg")
futures = client.scatter(files)
model_future = client.run(load_model)
results = client.map(run, futures, model = model_future)
