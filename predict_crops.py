from deepforest import deepforest
import pandas as pd
import glob
from crown_maps import start_cluster

client = start_cluster.start(gpus=5)

def run(paths):
    model = deepforest.deepforest()
    model.use_release()
    
    results = []
    for x in paths:
        boxes = model.predict_image(x, return_plot = False)
        boxes["file"] = x
        results.append(boxes)
    
    results = pd.concat(results)
    return results

files = glob.glob("/orange/ewhite/b.weinstein/NeonTreeEvaluation/pretraining/crops/*.jpg")
futures = client.scatter(files[:100])
results = client.submit(run, futures)

full_set = []
for x in results:
    result = results.result()
    full_set.append(result)

full_set = pd.concat(full_set)
full_set.to_csv("predictions.csv")
