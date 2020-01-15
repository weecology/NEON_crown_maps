from .. start_cluster import start_dask_cluster
import platform
import pytest
import dask
import time
import random

@pytest.fixture
def test_platform():
    test_platform = platform.system()
    return test_platform

def inc(x):
    time.sleep(random.random())
    return x + 1

def dec(x):
    time.sleep(random.random())
    return x - 1
    
def add(x, y):
    time.sleep(random.random())
    return x + y

def test_start_dask_cluster(test_platform):
    
    if test_platform =="Darwin":
        pass
    else:  
        client = start_dask_cluster(number_of_workers=2, mem_size="11GB")
        inc = dask.delayed(inc)
        dec = dask.delayed(dec)
        add = dask.delayed(add)
        
        zs = []
        for i in range(256):
            x = inc(i)
            y = dec(x)
            z = add(x, y)
            zs.append(z)
            
        zs = dask.persist(*zs)
        assert len(zs) == 256
        assert zs[0] == 0

