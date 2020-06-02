import geopandas as gpd
import pandas as pd
import rasterio
#from rasterstats import zonal_stats
from rasterio import features
from shapely.geometry import box
from fiona.crs import from_epsg
import fiona
import rasterio
import rasterio.mask
import re

def get_epsg(site):
    lookup = pd.read_csv("//orange/idtrees-collab/NLCD_2016/sites_utm_zone.csv")
    id = lookup['siteID']==site
    utmZone = lookup['utmZone'][id]
    utmZone=utmZone.iat[0]
    return(utmZone)


def getFeatures(gdf):
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def get_nlcd_clip(aitcput, outdir = "///orange/idtrees-collab/NLCD_2016/outdir/"):
    tile_corner = aitcput.split("_")
    try:
        #get boundaries of the tile from its name
        minx, miny =  int(aitcput.split("_")[3]),  int(aitcput.split("_")[4])
        maxx, maxy =  int(aitcput.split("_")[3])+1000,  int(aitcput.split("_")[4])+1000
        bbox = box(minx, miny, maxx, maxy)
        geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(get_epsg(aitcput.split("_")[1])))
        #load NLCD map
        nlcdpath = "///orange/idtrees-collab/NLCD_2016//NLCD_2016_Land_Cover_L48_20190424.img"
        # geo transform the polygon clip  
        with rasterio.open(nlcdpath) as src:
            out_meta = src.meta
            out_crs = src.crs
        #
        geo = geo.to_crs(crs = out_crs)
        geo = getFeatures(geo)
        #
        with rasterio.open(nlcdpath) as src:
            out_image, out_transform = rasterio.mask.mask(src, geo, crop = True)    
        #    
        #
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        #
        #save raater
        with rasterio.open(outdir+aitcput.split("_")[1]+"_"+aitcput.split("_")[2]+"_"+aitcput.split("_")[3]+"_"+aitcput.split("_")[4]+"_nlcd.tif", "w", **out_meta) as dest:
            dest.write(out_image)
        #
    except:
        print("error" + aitcput.split("_")[1]+"_"+aitcput.split("_")[2]+"_"+aitcput.split("_")[3]+"_"+aitcput.split("_")[4])


