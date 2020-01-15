from .. start_cluster import start_dask_cluster
import platform
import pytest
import dask
from dask.distributed import Client
import time
import random

@pytest.fixture
def test_platform():
    test_platform = platform.system()
    return test_platform

def inc(x):
    time.sleep(random.random())    
    return x + 1

def double(x):
    time.sleep(random.random())    
    return x + 2

def add(x, y):
    time.sleep(random.random())    
    return x + y

def test_start_dask_cluster(test_platform):
    
    if test_platform =="Darwin":
        client = Client()
        print(client)
        data = [1, 2, 3, 4, 5] * 10
        
        output = []
        for x in data:
            a = dask.delayed(inc)(x)
            b = dask.delayed(double)(x)
            c = dask.delayed(add)(a, b)
            output.append(c)
        
        output = dask.compute(*output)
        assert output[0] == 5
        assert len(output) ==50
        
    else:  
        client = start_dask_cluster(number_of_workers=2, mem_size="11GB")
        data = [1, 2, 3, 4, 5] * 10
        
        output = []
        for x in data:
            a = dask.delayed(inc)(x)
            b = dask.delayed(double)(x)
            c = dask.delayed(add)(a, b)
            output.append(c)
        
        output = dask.compute(*output)
        assert output[0] == 5
        assert len(output) ==50

