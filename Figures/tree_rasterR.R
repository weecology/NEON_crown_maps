library(sf)
library(raster)
library(stringr)

calc_density<-function(path,res=100){
  polygons <- read_sf(path)
  points<-st_centroid(polygons)
  r<-raster(polygons,res=c(res,res))
  points$mask <-1
  grid_density<-rasterize(points,r,field="mask",fun="sum")
  return(grid_density)
}

find_files<-function(site, year="2019", dir="/orange/idtrees-collab/draped/"){
  f<-list.files(dir,pattern=".shp", full.names = T)
  f<-f[str_detect(f,site)]
  f<-f[str_detect(f,year)]
  return(f)
}


tree_density<-function(site,year,dir="/orange/idtrees-collab/draped/", outdir="/orange/idtrees-collab/tree_density/"){
  paths<-find_files(site,year, dir)
  rasters <-lapply(paths,calc_density)
  merged_raster<-do.call(raster::merge,rasters)
  fn <- paste(outdir,site,".tif")
  writeRaster(merged_raster, fn)
}

