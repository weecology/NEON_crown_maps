#tests for analysis.py
import pytest
import glob
import numpy as np
import re

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from analysis import temporal


@pytest.fixture()
def geo_indexes():
    shps = glob.glob("data/temporal/*.shp")
    geo_indexes = [re.search("(\d+_\d+)_image",x).group(1) for x in shps]
    
    #Unique indexes
    geo_indexes = list(np.unique(np.array(geo_indexes)))
    
    return geo_indexes

#find unique geo_index
def test_match_years(geo_indexes, tmpdir):
    shps = glob.glob("data/temporal/*.shp")    
    for geo_index in geo_indexes:
        temporal.match_years(geo_index, shps,savedir=tmpdir, debug=True)
        saved_shps = glob.glob("{}/*.shp".format(tmpdir))
        assert len(saved_shps) == 2
    
def test_tree_fall(geo_indexes):
    shps = glob.glob("data/analysis/*.shp")    
    CHMs = glob.glob("data/analysis/*_CHM.tif")    
    for geo_index in geo_indexes:
        temporal.tree_falls(geo_index, shps, CHMs,savedir="output/")
    