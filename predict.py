from deepforest import deepforest
import pandas as pd
import dask

def create_model():
    model = deepforest.deepforest()
    model.use_release()
    return model

#predict a tile
def predict_tiles(tile_list, client=None):
    """Parallel loop through tile list and predict tree crowns
    Args:
        tile_list: a list of paths on disk of RGB tiles to predict
        client: a dask client with workers waiting for tasks (optional)
    """
    
    if client:
        #Create the model on each worker in the client
        model = dask.delayed(create_model())
        results = []
        for tile in tile_list:
            boxes = dask.delayed(model.predict_tile)(tile)
            results.append(boxes)
            
        all_boxes = dask.compute(*results)
        df = pd.concat(all_boxes)
        
    else:
        #Create the model on each worker in the client
        model = create_model()
        results = []
        for tile in tile_list:
            boxes = model.predict_tile(tile)
            results.append(boxes)
            
        all_boxes = results
        df = pd.concat(all_boxes)

    return df
