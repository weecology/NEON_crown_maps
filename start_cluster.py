"""
Create a cluster of GPU nodes to perform parallel prediction of tiles
"""
import argparse
from dask_utility import start_dask_cluster
from dask.distributed import wait

import sys
import socket
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, wait
import yaml

def args():
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--debug', help='Run local version without GPU', action='store_true')
    parser.add_argument('--workers', help='Number of dask workers', default="4")
    parser.add_argument('--memory_worker', help='GB memory per worker', default="10")

def find_tiles():
    """Read a yaml describing which sites to run"""
    pass

def start_tunnel():
    """
    Start a juypter session and ssh tunnel to view task progress
    """
    host = socket.gethostname()        
    print("To tunnel into dask dashboard:")
    print("ssh -N -L 8787:%s:8787 -l b.weinstein hpg2.rc.ufl.edu" % (host))
    
    #flush system
    sys.stdout.flush()
    
def start_dask_cluster(number_of_workers, mem_size="10GB"):
    #################
    # Setup dask cluster
    #################

    #job args
    extra_args=[
        "--error=/home/b.weinstein/logs/dask-worker-%j.err",
        "--account=ewhite",
        "--output=/home/b.weinstein/logs/dask-worker-%j.out"
    ]

    cluster = SLURMCluster(
        processes=1,
        queue='hpg2-compute',
        cores=1, 
        memory=mem_size, 
        walltime='24:00:00',
        job_extra=extra_args,
        local_directory="/home/b.weinstein/logs/dask/", death_timeout=300)

    print(cluster.job_script())
    cluster.adapt(minimum=number_of_workers, maximum=number_of_workers)

    dask_client = Client(cluster)

    #Start dask
    dask_client.run_on_scheduler(start_tunnel)  
    
    return dask_client

if __name__=="__main__":
    #Local debug. If False, paths on UF hypergator supercomputing cluster
    system_args = args()
    DEBUG = system_args.debug
  
    if DEBUG:
        dask_client = None
        tile_list = ""
    else:
        dask_client = start_dask_cluster(number_of_workers=system_args.workers, mem_size="11GB")
    
        #find list of tiles
        tile_list = find_tiles()
    
    df = predict_tiles(dask_client, tile_list)
    df.to_csv("predicted_trees.csv",index=False)
                
                
    

    
