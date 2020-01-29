from dask_jobqueue import SLURMCluster
from dask.distributed import Client

#job args
extra_args=[
    "--error=/home/b.weinstein/logs/dask-worker-%j.err",
    "--account=ewhite",
    "--output=/home/b.weinstein/logs/dask-worker-%j.out",
    "--partition=gpu",
    "--gpus=1",
 "module load tensorflow/1.14.0"]

cluster = SLURMCluster(
    processes=1,
    cores=1, 
    memory="20GB", 
    walltime='24:00:00',
    job_extra=extra_args,
    local_directory="/orange/ewhite/b.weinstein/NEON/logs/dask/", death_timeout=300)

print(cluster.job_script())
cluster.scale(2)    

client = Client(cluster)

#available
def available():
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