#dask worker model test
from start_cluster import start
from distributed import wait
import dask

client = start(gpus=2)

def load_model():
    from deepforest import deepforest
    model = deepforest.deepforest()
    model.use_release()
    return(model)

def run_test(model):
    from deepforest import get_data            
    path = get_data("OSBS_029.jpg")
    result = model.predict_image(path)
    return(result)

results = [ ]
for x in [1,2,3]:
    model = dask.delayed(load_model)()
    futures = dask.delayed(run_test)(model)
    results.append(futures)
    
results = dask.compute(*results)

wait(results)

for future in results:
    future.result()
