#Generate tfrecords
from start_cluster import start_dask_cluster
from dask.distributed import wait
from utils import tfrecords
import glob

def find_files():
    #tile_list = glob.glob("tests/data/*.tif")
    tile_list = glob.glob("/orange/ewhite/NeonData/**/*image.tif",recursive=True)
    return tile_list

#Start SLURM cluster
client = start_dask_cluster(number_of_workers=150)

#Find files
tile_list = find_files()

written_records = client.map(tfrecords.create_tfrecords, tile_list, patch_size=400, savedir="/orange/ewhite/b.weinstein/NEON/crops/")
wait(written_records)
print("{} records created".format(len(written_records)))
