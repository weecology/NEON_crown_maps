#Generate tfrecords
import glob
import random

from crown_maps.start_cluster import start
from dask.distributed import wait
from crown_maps import tfrecords


def find_files():
    #tile_list = glob.glob("tests/data/*.tif")
    tile_list = glob.glob("/orange/ewhite/NeonData/**/*image.tif",recursive=True)
    return tile_list

if __name__=="__main__":
    #Start SLURM cluster
    client = start(cpus=50, mem_size="7GB")
    
    #Find files
    tile_list = find_files()
    random.shuffle(tile_list)
    tile_list = tile_list[:50]
    
    written_records = client.map(tfrecords.create_tfrecords, tile_list, patch_size=400, savedir="/orange/ewhite/b.weinstein/NEON/crops/")
    wait(written_records)
    print("{} records created".format(len(written_records)))
