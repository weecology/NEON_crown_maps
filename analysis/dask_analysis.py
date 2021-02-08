#Analysis
from crown_maps.start_cluster import start
import glob
from analysis import temporal
import re
import numpy as np
from distributed import Client, as_completed, wait

debug = False
if debug:
    shps = glob.glob("/Users/ben/Dropbox/Weecology/Crowns/examples/*.shp")
    CHMs = glob.glob("/Users/ben/Dropbox/Weecology/Crowns/examples/*_CHM.tif")    
    savedir = "/Users/ben/Dropbox/Weecology/Crowns/"
    client = Client(processes=False)
else:
    #Start cluster
    client = start(cpus=20)
    shps = glob.glob("/orange/idtrees-collab/draped/*.shp")
    CHMs = glob.glob("/orange/ewhite/NeonData/**/CanopyHeightModelGtif/*.tif",recursive=True)
    savedir = "/orange/idtrees-collab/"

#Get geoindex
#Limit to OSBS
shps = [x for x in shps if "OSBS" in x]
geo_index = [re.search("(\d+_\d+)_image",x).group(1) for x in shps]
geo_index = np.unique(geo_index)

year_futures = client.map(temporal.match_years, geo_index, shps=shps,savedir=savedir + "growth/")
falls_futures = client.map(temporal.tree_falls, geo_index, shps=shps, CHMs=CHMs, savedir=savedir + "treefall/")

year_results = []
for future in as_completed(year_futures):
    try:
        result = future.result()
        year_results.append(result)
    except Exception as e:
        print("Year futures: {} failed with {}".format(future, e))   

fall_results = []
for future in as_completed(falls_futures):
    try:
        result = future.result()
        fall_results.append(result)
    except Exception as e:
        print("Tree futures: {} failed with {}".format(future, e))   

wait(year_futures)    
wait(falls_futures)
