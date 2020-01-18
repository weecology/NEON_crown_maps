#Generate tfrecords
from start_cluster import start_dask_cluster
import tfrecords

def find_files():
    tile_list = glob.glob("tests/data/*.tif")
    return tile_list

#Start SLURM cluster
client = start_dask_cluster(number_of_workers=2)

#Find files
tile_list = find_files()

written_records = client.map(tile_list, tfrecords.create_tfrecords, patch_size=800, savedir="/orange/ewhite/b.weinstein/NEON/crops/")
print("{} records created".format(len(written_records)))