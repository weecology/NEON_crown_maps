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
    #Heights and Areas
    sumstats = {"height":["mean","count","std"], "area":["mean",upper,"count","std"]}    
    average_height_area = daskdf.groupby(['Site']).agg(sumstats).compute().reset_index()
    average_height_area.columns = average_height_area.columns.map('_'.join)
    average_height_area = average_height_area.rename(columns={"Site_":"Site"})
    average_height_area = average_height_area.reset_index()
    
    #Number of trees
    average_density = daskdf.groupby(["Site","geo_index","Year"]).count().groupby("Site").left.mean().compute().reset_index()
    average_density = average_density.rename(columns = {"left":"n"})
        
    #Combine 
    results = average_height_area.merge(average_density)
    
    return results
    
if __name__ == "__main__":
    
    #Create dask client
    client = start_client(debug=False)
    
    #Create dataframe and compute summary statistics
    daskdf = load_predictions("/orange/ewhite/b.weinstein/NEON/draped/")
    
    #How many records total?
    total_trees = daskdf.shape[0].compute()
    total_sites = daskdf.Site.nunique().compute()
    
    print("There are {} tree predictions from {} sites".format(total_trees, total_sites))
    results = averages(daskdf)
    results.to_csv("Figures/averages.csv")
    
    #Count totals
    ntiles = daskdf.groupby(["Site","geo_index","Year"]).value_counts()
    ntiles.to_csv("Figures/counts.csv")
    
    
    
    