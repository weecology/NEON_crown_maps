import os
import rasterio
import argparse
from PIL import Image
import subprocess
import pathlib
import shutil
from glob import glob
from numba import njit, prange

from OpenVisus import *

### Configuration
ext_name = ".tif"
dtype = "uint8[3]"
limit = 1000
###--------------

@njit(parallel=True)
def blend_rgb_ann(a, b):
  #a[b[b>0]] = [255,0,0]
  for i in prange(a[0].shape[0]):
    for j in prange(a[0].shape[1]):
        if(b[i][j] > 0):
          a[0][i][j]=255
          a[1][i][j]=0
          a[2][i][j]=0

class tile():
  def __init__(self,path,name):
    self.path = path
    self.name = name
    self.frame = [0,0,0,0]
    self.size  = [0,0]

parser = argparse.ArgumentParser(description='Parse set of geotiff')
parser.add_argument('-rgb', type=str, nargs = 1, help ='rbg image path', required = True)
parser.add_argument('-ann', type=str, nargs = 1, help ='ann image path', required = False)
parser.add_argument('-out', type=str, nargs = 1, help ='output name', required = True)

args = parser.parse_args()

rgb_dir = args.rgb[0]
outdir = args.out[0]

pathlib.Path(outdir+"/temp").mkdir(parents=True, exist_ok=True)

outname = outdir.split("/")[-1]
if(outname==""):
  outname = outdir.split("/")[-2]

if(args.ann):
  ann_dir = args.ann[0]
  # Blend rgb and annotations

  for f in os.listdir(rgb_dir):
    if f.endswith(ext_name):
      rgb_path=rgb_dir+"/"+f
      ann_path=ann_dir+"/"+f.replace("image.tif", "image_rasterized.tif")
      
      ageo = rasterio.open(rgb_path)
      a = ageo.read()
      bgeo = rasterio.open(ann_path)
      b = bgeo.read()
      print("Blending ", rgb_path, "and", ann_path, "...")
      blend_rgb_ann(a, b[0])
      #tiff.imsave(outdir+"/"+f,a)

      with rasterio.open(
          outdir+"/"+f,
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

    idir = outdir
else:
  idir = rgb_dir

# Convert and stitch
images = []

for f in os.listdir(idir):
  if f.endswith(ext_name):
    filepath=idir+"/"+f
    s = os.path.basename(f)
    # filepath = filepath.replace('(','\(')
    # filepath = filepath.replace(')','\)')
    images.append(tile(filepath,s))

bbox = [99999999, 0, 99999999, 0]

count = 0
for img in images:
  if count > limit:
    break
  count += 1

  try:
    ds = rasterio.open(img.path)
    width = ds.width
    height = ds.height
    bounds = ds.bounds
  except:
    print("ERROR: metadata failure, skipping "+idir)
    
  minx = bounds.left 
  miny = bounds.top 
  maxx = bounds.right 
  maxy = bounds.bottom

  img.frame = [minx, maxx, miny, maxy]
  img.size = [width, height]
  #print("found gdal data", gt, "size", [height, width], "frame", [minx, maxx, miny, maxy], "psize", [maxx-minx, maxy-miny])
  print("frame", img.frame)#, "psize", [(maxx-minx)/width, (maxy-miny)/height])

  if(minx < bbox[0]):
    bbox[0] = minx

  if(miny < bbox[2]):
    bbox[2] = miny

  if(maxx > bbox[1]):
    bbox[1] = maxx

  if(maxy > bbox[3]):
    bbox[3] = maxy

ratio=[(maxx-minx)/width,(maxy-miny)/height]

out_size = [bbox[1]-bbox[0], bbox[3]-bbox[2]]
img_size = [int(out_size[0]/ratio[0]), int(out_size[1]/ratio[1])]

gbox = "0 "+str(img_size[0]-1)+" 0 "+str(img_size[1]-1)
midx_name=outdir+"/global.midx"
midx_out = open(midx_name,"wt")
midx_out.write("<dataset typename='IdxMultipleDataset'>\n")
midx_out.write('<field name="voronoi">\n  <code>output=voronoi()</code>\n</field>')

cwd = os.getcwd()

count = 0
for img in images:
  if count > limit:
    break
  count += 1

  lbox = "0 "+str(img.size[0]-1)+" 0 "+str(img.size[1]-1)
  ancp = [int((img.frame[0]-bbox[0])/ratio[0]), int((img.frame[2]-bbox[2])/ratio[1])]
  #print(ancp)
  dbox = str(ancp[0])+ " " +str(ancp[0]+img.size[0]-1)+ " "+str(ancp[1])+ " "+str(ancp[1]+img.size[1]-1)
  #midx_out.write('\t<dataset url="file://'+outdir+"/"+img.name+'exp.idx" name="'+img.name+'"> <M><translate x="'+str(ancp[0])+'" y="'+str(ancp[1])+'"/></M> </dataset>\n')
  midx_out.write('\t<dataset url="file://'+outdir+"/"+img.name+'exp.idx" name="'+img.name+'" offset="'+str(ancp[0])+' '+str(ancp[1])+'"/>\n')

  exp_idx = outdir+"/"+img.name+"exp.idx"

  field=Field("data",dtype,"row_major")
  CreateIdx(url=exp_idx,dims=img.size,fields=[field])
  db=PyDataset(exp_idx)

  #convertCommand(["create", exp_idx, "--box", lbox, "--fields", 'data '+dtype,"--time","0 0 time%03d/"])
  #convert.runFromArgs(["create", exp_idx, "--box", lbox, "--fields", 'data '+dtype,"--time","0 0 time%03d/"])

  print("Converting "+str(count)+"/"+str(min(limit, len(images)))+"...")

  data=numpy.asarray(Image.open(img.path))
  db.write(data)

  #convertCommand(["import",img.path,"--dims",str(img.size[0]),str(img.size[1])," --dtype ",dtype,"--export",exp_idx," --box ",lbox, "--time", "0"])
  #convert.runFromArgs(["import",img.path,"--dims",str(img.size[0]),str(img.size[1])," --dtype ",dtype,"--export",exp_idx," --box ",lbox, "--time", "0"])

midx_out.write('</dataset>')
midx_out.close();

print("Done conversion of tiles, now generating final mosaic")


def midxToIdx(filename, filename_idx):
  field="output=voronoi()"
  # in case it's an expression
  tile_size=int(eval("4*1024"))

  DATASET = LoadIdxDataset(filename)
  FIELD=DATASET.getFieldByName(field)
  TIME=DATASET.getDefaultTime()
  Assert(FIELD.valid())

  # save the new idx file
  idxfile=DATASET.idxfile
  idxfile.filename_template = "" # //force guess
  idxfile.time_template = ""     #force guess
  idxfile.fields.clear()
  idxfile.fields.push_back(Field("DATA", dtype, "rowmajor")) # note that compression will is empty in writing (at the end I will compress)
  idxfile.save(filename_idx)

  dataset = LoadIdxDataset(filename_idx)
  Assert(dataset)
  field=dataset.getDefaultField()
  time=dataset.getDefaultTime()
  Assert(field.valid())

  ACCESS = DATASET.createAccess()
  access = dataset.createAccess()

  print("Generating tiles...",tile_size)
  TILES = DATASET.generateTiles(tile_size)
  TOT_TILES=TILES.size()
  T1 = Time.now()
  for TILE_ID in range(TOT_TILES):
    TILE = TILES[TILE_ID]
    t1 = Time.now()
    buffer = DATASET.readFullResolutionData(ACCESS, FIELD, TIME, TILE)
    msec_read = t1.elapsedMsec()
    if not buffer: 
      continue

    t1 = Time.now()
    dataset.writeFullResolutionData(access, field, time, buffer, TILE)
    msec_write = t1.elapsedMsec()

    print("done", TILE_ID, "/", TOT_TILES, "msec_read", msec_read, "msec_write", msec_write)

  #dataset.compressDataset("jpg-JPEG_QUALITYGOOD-JPEG_SUBSAMPLING_420-JPEG_OPTIMIZE")
  #dataset.compressDataset("jpg-JPEG_QUALITYSUPERB-JPEG_SUBSAMPLING_420-JPEG_OPTIMIZE")
  dataset.compressDataset("jpg-JPEG_QUALITYSUPERB-JPEG_SUBSAMPLING_444-JPEG_OPTIMIZE")
  #dataset.compressDataset("jpg-JPEG_QUALITYGOOD-JPEG_SUBSAMPLING_444-JPEG_OPTIMIZE")

# Make one big photomosaic
midxToIdx(os.path.abspath(midx_name), os.path.abspath(outdir+"/"+outname+".idx"))

# moving clutter to "outdir/temp" folder
for f in glob.glob(outdir+"/*tifexp*"):
  subprocess.run(["mv",f,outdir+"/temp/"])
for f in glob.glob(outdir+"/*.tif"):
  subprocess.run(["mv",f,outdir+"/temp/"])
subprocess.run(["mv",outdir+"/global.midx",outdir+"/temp/"])

# delete temp folder at the end
#subprocess.run(["rm","-R", outdir+"/temp"])

print("DONE")

