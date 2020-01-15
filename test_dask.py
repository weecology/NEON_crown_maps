from start_cluster import start_dask_cluster
import glob
import predict

client = start_dask_cluster(number_of_workers=2, mem_size="11GB")
boxes = predict.predict_tiles(tile_list = glob.glob("tests/data/*.tif"), client=client)
assert boxes.shape[1] == 6
