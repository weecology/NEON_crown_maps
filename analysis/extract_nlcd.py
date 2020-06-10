import geopandas as gpd
import pandas as pd
import os
import glob
import rasterio
import rasterstats 
from distributed import wait

from scipy.stats import mode
from shapely.geometry import Point
from rasterio.crs import CRS
from crown_maps import start_cluster

def get_epsg(site):
    lookup = pd.read_csv("//orange/idtrees-collab/NLCD_2016/sites_utm_zone.csv")
    utmZone = lookup[lookup.siteID == site].utmZone.values[0]
    
    return utmZone

def create_point(x,y):
    
    return Point(x,y)

def raster_mode(x):
    """Get height quantile of all cells that are no zero"""
    
    return mode(x)[0][0][0]

def run(site_csv):
    print(site_csv)
    #read file
    sitedf = pd.read_csv(site_csv)
    
    #Create point class 
    sitedf["geometry"] = sitedf.apply(lambda x: create_point(x.plot_center_x, x.plot_center_y),axis=1)
    site = sitedf.site.unique()[0]
    
    #Get spatial projection
    epsg = get_epsg(site)
    geodf = gpd.GeoDataFrame(sitedf,geometry="geometry",crs=CRS.from_epsg(get_epsg(site)))
    geodf.head()
    print(geodf.crs)

    #Load NLCD
    nlcdpath = "/orange/idtrees-collab/NLCD_2016/NLCD_2016_Land_Cover_L48_20190424.img"
    
    # geo transform the polygon clip
    with rasterio.open(nlcdpath) as src:
        out_meta = src.meta
        nlcd_crs = src.crs
    
    geodf = geodf.to_crs(src.crs)
    
    #extract nlcd class
    class_dict = rasterstats.zonal_stats(geodf, nlcdpath, stats="mean",add_stats={'mode':raster_mode})
    sitedf["nlcd"]  = [g["mode"] for g in class_dict]
    
    #write alonside original
    fn = "{}_nlcd.csv".format(os.path.splitext(site_csv)[0])
    sitedf.to_csv(fn)
    
if __name__ == "__main__":
    #client = start_cluster.start(cpus=10)
    file_list = glob.glob("/home/b.weinstein/NEON_crown_maps/Figures/sampling*.csv")
    for fn in file_list:
        try:
           	run(fn)
	except:
		pass
    #futures = client.map(run, file_list)
    #wait(futures)
    
