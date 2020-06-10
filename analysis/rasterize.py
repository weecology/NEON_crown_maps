#rasterize predictions
import geopandas as gp
import os
import rasterio
from rasterio import features
from shapely.geometry import box
from crown_maps import start_cluster
import glob
from distributed import wait

def read_shapefile(path):
    shapefile = gp.read_file(path)
    
    return shapefile

def find_rgb_path(path, rgb_dir=None):
    
    #Search rgb path recursive for paths
    rgb_pool = glob.glob(os.path.join(rgb_dir + "**/*.tif"),recursive=True)
    basename = os.path.basename(os.path.splitext(path)[0])
    
    rgb_match = [x for x in rgb_pool if basename in x] 
    
    if not len(rgb_match) == 1:
        raise IOError("Cannot find matching RGB file for {}".format(basename))
    
    return rgb_match[0]

def read_rgb(rgb_path):
    rst = rasterio.open(rgb_path)
    return rst

def rasterize_shapefile(shapes,rst,out_fn):
    """shapes: geopandas dataframe
        rst: open raster object to use bounds
        out_fn: outputfile name
    """
    
    meta = rst.meta.copy()
    meta["count"]=1
    meta["nodata"]=0
    print(meta)
    #Output filename
    with rasterio.open(out_fn, 'w+', **meta) as out:
        out_arr = out.read(1)
    
        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes["color"] = 1
        boxes = [x.exterior for x in shapes.geometry]
        shapes = ((geom,value) for geom, value in zip( boxes,shapes.color))
    
        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write_band(1, burned)
    
    
def run(path,rgb_dir=".",savedir="."):
    """path: path to shapefile"""
    
    shapefile = read_shapefile(path)
    
    rgb_path = find_rgb_path(path,rgb_dir)      
    
    rst = read_rgb(rgb_path)
    
    #Create output filename
    out_fn = "{}_rasterized.tif".format(os.path.splitext(os.path.basename(path))[0])
    out_fn = os.path.join(savedir,out_fn)
    rasterize_shapefile(shapefile, rst, out_fn)
    
if __name__ =="__main__":
    
    client = start_cluster.start(cpus=30, mem_size="6GB")
    
    #list files
    tiles_to_rasterize = glob.glob("/orange/idtrees-collab/draped/*.shp")
    
    print("Found {} tiles to rasterize".format(len(tiles_to_rasterize)))
    
    #apply raster function
    futures = client.map(run, tiles_to_rasterize, rgb_dir = "/orange/ewhite/NeonData/", savedir="/orange/idtrees-collab/rasterized/")
    
    wait(futures)
    
    
