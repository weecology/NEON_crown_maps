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
from numba import njit, prange

from OpenVisus.__main__ import MidxToIdx

def match_name(x):
  x = os.path.basename(x)
  return x.replace("image.tif","image_rasterized.tif")

@njit(parallel=True)
def blend_rgb_ann(a, b):
  #a[b[b>0]] = [255,0,0]
  for i in prange(a[0].shape[0]):
    for j in prange(a[0].shape[1]):
      if(b[i][j] > 0):
        a[0][i][j]=255
        a[1][i][j]=0
        a[2][i][j]=0
        
def blend(rgb_path, annotation_dir, outdir):
  basename = os.path.basename(rgb_path)
  ann_path=annotation_dir+"/"+basename.replace("image.tif", "image_rasterized.tif")
  ageo = rasterio.open(rgb_path)
  a = ageo.read()
  bgeo = rasterio.open(ann_path)
  b = bgeo.read()
  print("Blending ", rgb_path, "and", ann_path, "...")
  blend_rgb_ann(a, b[0])
  out_name = outdir+"/"+basename
  with rasterio.open(
        out_name,
        'w',
        driver='GTiff',
        height=ageo.height,
        width=ageo.width,
        count=3,
        dtype=a.dtype,
        crs='+proj=latlong',
        transform=ageo.transform,
    ) as dst:
        dst.write(a)
  
  return out_name
  
def run(rgb_images, dst_directory, annotation_dir):
  
  #Construct outdir variable from top level savedir and site
  site = get_site(rgb_images[0])  
  outdir = os.path.join(dst_directory,site)
  pathlib.Path(outdir+"/temp").mkdir(parents=True, exist_ok=True)
  
  outname = outdir.split("/")[-1]
  if(outname==""):
    outname = outdir.split("/")[-2]
  
  # Blend rgb and annotations
  images = []
  for rgb_path in rgb_images:
    images.append(blend(rgb_path, annotation_dir, outdir))
    
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
  idx_filename=os.path.join(dst_directory,"visus.idx")
  MidxToIdx(["--midx", midx_filename, "--field","output=voronoi()", "--tile-size","4*1024", "--idx", idx_filename])
  

if __name__=="__main__":  
  #Create dask cluster
  from crown_maps import start_cluster
  client = start_cluster.start(cpus=20,mem_size="40GB")
  client.wait_for_workers(1)
    
  #Pool of rasterized predictions
  rgb_list = glob.glob("/orange/ewhite/NeonData/**/Mosaic/*image.tif",recursive=True)  
  annotation_dir = "/orange/idtrees-collab/rasterized/"
  outdir = "/orange/idtrees-collab/OpenVisus/"
  annotation_list = glob.glob(annotation_dir + "*.tif")
  
  annotation_names = [os.path.basename(x) for x in annotation_list]
  rgb_list = [x for x in rgb_list if match_name(x) in annotation_names]
  
  df = pd.DataFrame({"path":rgb_list})
  df["site"] = df.path.apply(lambda x: get_site(x))
  df["year"] = df.path.apply(lambda x: get_year(x))
  
  #just run
  df = df[df.site.isin(["ABBY"])]
  
  #order by site  using only the most recent year
  site_lists = df.groupby('site').apply(lambda x: x[x.year==x.year.max()]).reset_index(drop=True).groupby('site').path.apply(list).values
  
  ####Scatter and run in parallel
  futures = []
  for site in site_lists:
    site = np.sort(site)
    siteID = get_site(site[0])
    site_dir = "{}/{}".format(outdir, siteID)
    
    try:
      shutil.rmtree(site_dir)
      os.mkdir(site_dir) 
    except:
      os.mkdir(site_dir)       
        
    future = dask.delayed(run)(rgb_images=site[0:20], dst_directory=site_dir, annotation_dir=annotation_dir)
    futures.append(future)
    
  persisted_values = dask.persist(*futures)
  distributed.wait(persisted_values)
  for pv in persisted_values:
    try:
      print(pv)
    except Exception as e:
      print(e)
      continue  