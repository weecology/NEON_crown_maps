#test figures
#relative path hack just for pytest
import os
import sys
import glob
sys.path.append(os.path.dirname(os.getcwd()))

from .. import figures
import pytest

@pytest.fixture()
def client():
 client = figures.start_client(debug=True)
 return client

@pytest.fixture()
def daskdf(client):
 daskdf = figures.load_predictions("data/")
 return daskdf

def test_load_shp():
 for shp in glob.glob("data/" + "*.shp"):
  df = figures.load_shp(shp)
  assert not df.empty
 
def test_load_predictions(client):
    daskdf = figures.load_predictions("data/")
    assert len(daskdf.Site.unique().compute()) == 1
    
def test_averages(client,daskdf):
    results = figures.averages(daskdf)
    assert results.shape == (1,4)