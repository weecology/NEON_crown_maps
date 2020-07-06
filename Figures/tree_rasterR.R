library(sf)
library(raster)
library(stringr)

calc_density<-function(path,res=100, outdir="/orange/idtrees-collab/tree_density/"){
  print(path)
  polygons <- read_sf(path)
  points<-st_centroid(polygons)
  r<-raster(polygons,res=c(res,res))
  points$mask <-1
  grid_density<-rasterize(points,r,field="mask",fun="sum")
  fn <- str_match(path,"/(\\w+).shp")[,2]
  fn <- paste(outdir,fn,".tif",sep = "")
  writeRaster(grid_density, fn, overwrite=T)
  
  return(fn)
}

find_files<-function(site, year="2019", dir="/orange/idtrees-collab/draped/"){
  f<-list.files(dir,pattern=".shp", full.names = T)
  f<-f[str_detect(f,site)]
  f<-f[str_detect(f,year)]
  return(f)
}


tree_density<-function(site,year,dir="/orange/idtrees-collab/draped/"){
  paths<-find_files(site,year, dir)
   lapply(paths,calc_density)
}

tree_density("TEAK","2019")
