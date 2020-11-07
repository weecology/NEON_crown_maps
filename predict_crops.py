from deepforest import deepforest
import pandas as pd
import glob
from crown_maps import start_cluster

client = start_cluster.start(gpus=5)

def run(paths):
    saved_model="/home/b.weinstein/miniconda3/envs/DeepTreeAttention_DeepForest/lib/python3.7/site-packages/deepforest/data/NEON.h5"
    model = deepforest.deepforest(saved_model = saved_model)
    
    results = []
    for x in paths:
        boxes = model.predict_image(x, return_plot = False)
        boxes["file"] = x
        results.append(boxes)
    
    results = pd.concat(results)
    return results

files = glob.glob("/orange/ewhite/b.weinstein/NeonTreeEvaluation/pretraining/crops/*.jpg")
futures = client.scatter(files[:100])
results = client.map(run, futures)
results = client.gather(results)
full_set = pd.concat(full_set)
full_set.to_csv("predictions.csv")
