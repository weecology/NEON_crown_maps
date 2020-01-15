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

def test_dask(test_platform):
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
