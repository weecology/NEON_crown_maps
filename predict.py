import deepforest
import pandas as pd
import dask

def create_model():
    model = deepforest.deepforest()
    model.use_release()
    return model

#predict a tile
def predict_tiles(tilelist, dask_client=None):
    """Parallel loop through tile list and predict tree crowns
    Args:
        tilelist: a list of paths on disk of RGB tiles to predict
        dask_client: a dask client with workers waiting for tasks (optional)
    """
    
    #Create the model on each worker in the client
    results = []
    for tile in tilelist:
        model = dask.delayed(create_model)
        boxes = model.predict_tile(tile)
        results.append(boxes)
        
    all_boxes = dask.compute(results)
    df = pd.concat(all_boxes)
    print(df.head())
    print("Data frame has shape {}".format(df.shape))
    
    return df
