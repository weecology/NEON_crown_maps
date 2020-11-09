import pandas as pd
import glob
import numpy as np
from crown_maps import start_cluster

client = start_cluster.start(gpus=5)
client.wait_for_workers(2)

def run(paths):
    from deepforest import deepforest    
    saved_model="/home/b.weinstein/miniconda3/envs/DeepTreeAttention_DeepForest/lib/python3.7/site-packages/deepforest/data/NEON.h5"
    model = deepforest.deepforest(saved_model = saved_model)
    
    for x in paths:
        boxes = model.predict_image(x, return_plot = False)
        boxes["file"] = x
        csv_name = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/pretraining/predictions/{}.csv".format(x)
        boxes.to_csv(csv_name)

files = glob.glob("/orange/ewhite/b.weinstein/NeonTreeEvaluation/pretraining/crops/*.jpg")
chunks = np.array_split(np.array(files),5)
futures = client.scatter(chunks)
model_future = client.run(load_model)
results = client.map(run, futures, model = model_future)
client.gather(results)