"""
OpenVisus conversion module
"""
import os
import rasterio
import argparse
import subprocess
import pathlib
import shutil
from glob import glob
from PIL import Image,ImageOps

from OpenVisus import *
DbModule.attach()

import dask
import distributed
import pandas as pd
from crown_maps.verify import get_site, get_year
import numpy as np

def match_name(x):
  x = os.path.basename(x)
  return x.replace("image.tif","image_rasterized.tif")
def run(images, dst_directory):
  
  # find images
  # convert to idx
  sx,sy=1.0, 1.0
  tiles=[]
  for I,filename in enumerate(images):
    metadata =rasterio.open(filename)
    name=os.path.splitext(os.path.basename(filename))[0]
    width,height=metadata.width,metadata.height
    x1,y1,x2,y2=metadata.bounds.left,metadata.bounds.bottom, metadata.bounds.right, metadata.bounds.top
  
    # compute scaling to keep all pixels
    if I==0:
      sx=width /(x2-x1)
      sy=height/(y2-y1)
  
    x1,y1,x2,y2=sx*x1,sy*y1,sx*x2,sy*y2
    tile={"name": name, "size" : (width,height), "bounds" : (x1,y1,x2,y2)}
    print("Converting tile...",tile,I,"/",len(images))
    tiles.append(tile) 
  
    # avoid creation multiple times
    if not os.path.isfile(os.path.join(dst_directory,name,"visus.idx")):
      data=Image.open(filename)
      data=ImageOps.flip(data)
      CreateIdx(url=os.path.join(dst_directory,name,"visus.idx"), rmtree=True, dim=2,data=numpy.asarray(data))
  
  # create midx
  X1=min([tile["bounds"][0] for tile in tiles])
  Y1=min([tile["bounds"][1] for tile in tiles])
  midx_filename=os.path.join(dst_directory,"visus.midx")
  with open(midx_filename,"wt") as file:
    file.writelines([
            "<dataset typename='IdxMultipleDataset'>\n",
                  "\t<field name='voronoi'><code>output=voronoi()</code></field>\n",
                  *["\t<dataset url='./{}/visus.idx' name='{}' offset='{} {}'/>\n".format(tile["name"],tile["name"],tile["bounds"][0]-X1,tile["bounds"][1]-Y1) for tile in tiles],
                  "</dataset>\n"
          ])
  
  # to see automatically computed idx file
  db=LoadDataset(midx_filename)
  print(db.getDatasetBody().toString())
  
  from OpenVisus.__main__ import MidxToIdx
  idx_filename=os.path.join(dst_directory,"visus.idx")
  MidxToIdx(["--midx", midx_filename, "--field","output=voronoi()", "--tile-size","4*1024", "--idx", idx_filename])
  

if __name__=="__main__":  
  #Create dask cluster
  from crown_maps import start_cluster
  client = start_cluster.start(cpus=10,mem_size="40GB")
  client.wait_for_workers(1)
  
  #Pool of RGB images
  rgb_list = glob.glob("/orange/ewhite/NeonData/**/Mosaic/*image.tif",recursive=True)
  
  #Pool of rasterized predictions
  annotation_dir = "/orange/idtrees-collab/rasterized/"
  outdir = "/orange/idtrees-collab/OpenVisus/"
  annotation_list = glob.glob(annotation_dir + "*.tif")
  
  #filter names
  annotation_names = [os.path.basename(x) for x in annotation_list]
  rgb_list = [x for x in rgb_list if match_name(x) in annotation_names]
  
  df = pd.DataFrame({"path":rgb_list})
  df["site"] = df.path.apply(lambda x: get_site(x))
  df["year"] = df.path.apply(lambda x: get_year(x))
  
  #just run OSBS
  #df = df[df.site.isin(["ABBY"])]
  
  #order by site  using only the most recent year
  site_lists = df.groupby('site').apply(lambda x: x[x.year==x.year.max()]).reset_index(drop=True).groupby('site').path.apply(list).values
  
  ####Scatter and run in parallel
  futures = []
  for site in site_lists:
    site = np.sort(site)
    siteID = get_site(site[0])
    site_dir = "{}/{}".format(outdir, siteID)
    try:
      os.mkdir(site_dir)
    future = dask.delayed(run)(images=site, dst_directory=site_dir)
    futures.append(future)
    
  persisted_values = dask.persist(*futures)
  distributed.wait(persisted_values)
  for pv in persisted_values:
    try:
      print(pv)
    except Exception as e:
      print(e)
      continue  