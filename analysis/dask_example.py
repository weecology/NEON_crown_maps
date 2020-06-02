from dask_jobqueue import SLURMCluster
from dask.distributed import Client

#job args
extra_args=[
    "--error=/home/b.weinstein/logs/dask-worker-%j.err",
    "--account=ewhite",
    "--output=/home/b.weinstein/logs/dask-worker-%j.out",
    "--partition=gpu",
    "--gpus=1"]

cluster = SLURMCluster(
    processes=1,
    cores=1, 
    memory="20GB", 
    walltime='24:00:00',
    job_extra=extra_args,
    env_extra=['module load tensorflow/1.14.0',
               'export PATH=${PATH}:/apps/tensorflow/1.14.0py3/bin',
               'echo $PYTHONPATH',
               'echo $PATH'],
    local_directory="/orange/ewhite/b.weinstein/NEON/logs/dask/", death_timeout=300)

print(cluster.job_script())
cluster.scale(2)    

client = Client(cluster)

#available
def available():
    import os, sys
    sys.path.insert(0,"/apps/lmod/lmod/init/")
    
    from env_modules_python import module
    
    module("load","tensorflow/1.14")
    
    import tensorflow as tf    
    return tf.test.is_gpu_available()

#list devices
def devices():
    from tensorflow.python.client import device_lib
    return device_lib.list_local_devices()

#submit 
future = client.submit(devices)
print(future.result())
    
#submit 
future = client.submit(available)
print(future.result())