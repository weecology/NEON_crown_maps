from start_cluster import start_dask_cluster
import platform
import pytest
import dask
import time
import random

@pytest.fixture
def test_platform():
    test_platform = platform.system()
    return test_platform

@dask.delayed
def inc(x):
    time.sleep(random.random())    
    return x + 1

@dask.delayed
def double(x):
    time.sleep(random.random())    
    return x + 2

@dask.delayed
def add(x, y):
    time.sleep(random.random())    
    return x + y

client = start_dask_cluster(number_of_workers=2, mem_size="11GB")
data = [1, 2, 3, 4, 5] * 10

output = []
for x in data:
    a = inc(x)
    b = double(x)
    c = add(a, b)
    output.append(c)

output = dask.compute(output)[0]
assert output[0] == 5
assert len(output) ==50

