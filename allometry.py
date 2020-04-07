#Allometry
from sklearn.linear_model import LinearRegression
import geopandas  
import pandas as pd
import glob
import re
from dask import delayed
import dask.dataframe as dd
from check_site import get_site, get_year

def load_shp(shp):
    df = geopandas.read_file(shp)
    df["shp_path"] = shp
    df["geo_index"] = str(re.search("(\d+_\d+)_image",shp).group(1))
    df["Year"] = int(re.search("(\d+)_(\w+)_\d_\d+_\d+_image",shp).group(1))
    df["Site"] = re.search("(\d+)_(\w+)_\d_\d+_\d+_image",shp).group(2)
    df = df.drop(columns="geometry")
    return df
    
def load_predictions(tile_lists):
    """Load shapefiles from a path directory and convert into a persisted dask dataframe"""
    lazy_dataframes = []    
    for shp in tile_lists:
        gdf = delayed(load_shp)(shp)
        df = delayed(pd.DataFrame)(gdf)
        lazy_dataframes.append(df)
    
    daskdf = dd.from_delayed(lazy_dataframes, meta=lazy_dataframes[0].compute())
    df = daskdf.compute()
    return df

def fit_model(df):
    model = LinearRegression()
    X = df[["height"]]
    y = df[["area"]]    
    fit = model.fit(X,y)
    
    return fit
    
def run(tile_list):
    #Load data in parallel and read in DataFrame locally
    df = load_predictions(tile_list)
    model_fit = fit_model(df)
    
    #Format 
    data = {"slope": model_fit.coef_, "intercept":model_fit.intercept_, "min_height": df.height.min(),"max_height": df.height.quantile(0.99)}
    return data
    
if __name__ == "__main__":
    from start_cluster import start
    
    #Get pool of predictions    
    shps = glob.glob("/orange/idtrees-collab/draped/*.shp")
    
    #Dask cluster
    client = start(cpus=50)
    
    #Get site names
    df = pd.DataFrame({"path":shps})
    df["year"] = df.path.apply(lambda x: get_year(x))
    df["site"] = df.path.apply(lambda x: get_site(x))
    
    #Construct list of site+year combinations
    site_lists = df.groupby(['site','year']).path.apply(list).to_dict()
    
    #For each site year
    model_fit = {}
    for key, value in site_lists.items():
        model_fit[key] = run(value)
    
    modeldf = pd.DataFrame(model_fit)
    modeldf.to_csv("Figures/allometry.csv")
        

