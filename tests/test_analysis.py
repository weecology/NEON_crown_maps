#tests for analysis.py
from .. import analysis
import pytest
import glob
import numpy as np
import re

@pytest.fixture()
def geo_indexes():
    shps = glob.glob("data/*.shp")
    geo_indexes = [re.search("(\d+_\d+)_image",x).group(1) for x in shps]
    
    #Unique indexes
    geo_indexes = list(np.unique(np.array(geo_indexes)))
    
    return geo_indexes

#find unique geo_index
def test_match_years(geo_indexes):
    shps = glob.glob("data/*.shp")    
    for geo_index in geo_indexes:
        matched_df = analysis.match_years(geo_index, shps,savedir="/Users/ben/Dropbox/Weecology/Crowns/growth/")
    
def test_tree_fall(geo_indexes):
    shps = glob.glob("data/*.shp")    
    CHMs = glob.glob("data/*_CHM.tif")    
    for geo_index in geo_indexes:
        matched_df = analysis.tree_falls(geo_index, shps, CHMs,savedir="/Users/ben/Dropbox/Weecology/Crowns/treefall/")
    