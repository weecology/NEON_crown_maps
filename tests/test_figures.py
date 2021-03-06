#test figures
#relative path hack just for pytest
import os
import sys
import glob
sys.path.append(os.path.dirname(os.getcwd()))

from analysis import figures
import pytest

@pytest.fixture()
def client():
 client = figures.start_client(debug=True)
 return client

@pytest.fixture()
def daskdf(client):
 daskdf = figures.load_predictions("data/analysis/")
 return daskdf

def test_load_shp():
 for shp in glob.glob("data/analysis/" + "*.shp"):
  df = figures.load_shp(shp)
  df.groupby("Site").height.mean()
  assert not df.empty
 
def test_load_predictions(client):
    daskdf = figures.load_predictions("data/analysis/")
    assert len(daskdf.Site.unique().compute()) == 1
    
def test_site_averages(client,daskdf):
    results = figures.site_averages(daskdf)
    assert results.shape == (1,9)
 
def test_tile_averages(client,daskdf):
 results = figures.tile_averages(daskdf)
 assert results.shape == (2,9)
 
def test_counts(client, daskdf):
  ntiles = daskdf.groupby(["Site","geo_index","Year"]).size().compute()
  assert not ntiles.empty

#def test_treefalls(client):
  #results = figures.treefalls("/Users/ben/Dropbox/Weecology/Crowns/treefall/")
  #print(results)
  #assert not results.empty