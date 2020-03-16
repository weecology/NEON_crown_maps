"""dask module for creating figures from very large pandas frames of site predictions"""
import glob
import re
import geopandas
import pandas as pd
import dask.dataframe as dd

from start_cluster import start
from distributed import Client
from dask import delayed

def start_client(debug=True):
    if debug:
        client = Client()
    else:
        start(cpus=30)

def load_shp(shp):
    df = geopandas.read_file(shp)
    df["shp_path"] = shp
    df["geo_index"] = str(re.search("(\d+_\d+)_image",shp).group(1))
    df["Year"] = int(re.search("(\d+)_(\w+)_\d_\d+_\d+_image.shp",shp).group(1))
    df["Site"] = re.search("(\d+)_(\w+)_\d_\d+_\d+_image.shp",shp).group(2)
    df = df.drop(columns="geometry")
    return df
    
def load_predictions(path):
    """Load shapefiles from a path directory and convert into a persisted dask dataframe"""
    lazy_dataframes = []    
    for shp in glob.glob(path + "*.shp"):
        gdf = delayed(load_shp)(shp)
        df = delayed(pd.DataFrame)(gdf)
        lazy_dataframes.append(df)
    
    daskdf = dd.from_delayed(lazy_dataframes, meta=lazy_dataframes[0].compute())
    daskdf = daskdf.persist()
    return daskdf

def averages(daskdf):
    #Calculate average attributes
    average_height = daskdf.groupby(['Site']).height.mean().reset_index().compute()
    average_area = daskdf.groupby(["Site"]).area.mean().reset_index().compute()
    average_density = daskdf.groupby(["Site","geo_index","Year"]).count().groupby("Site").left.mean().reset_index().compute()
    average_density = average_density.rename(columns = {"left":"n"})
    
    results = average_height.merge(average_area)
    results = results.merge(average_density)
    return results

if __name__ == "__main__":
    
    #Create dask client
    client = start_client(debug=False)
    
    #Create dataframe and compute summary statistics
    daskdf = load_predictions("/orange/ewhite/b.weinstein/NEON/draped/")
    
    results = averages(daskdf)
    
    results.to_csv("Figures/averages.csv")
    
    
    
    