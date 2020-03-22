#Analysis
from start_cluster import start
import glob
import analysis
import re
import numpy as np
import pandas as pd
from distributed import Client, as_completed, wait

debug = False
if debug:
    shps = glob.glob("/Users/ben/Dropbox/Weecology/Crowns/examples/*.shp")
    CHMs = glob.glob("/Users/ben/Dropbox/Weecology/Crowns/examples/*_CHM.tif")    
    savedir = "/Users/ben/Dropbox/Weecology/Crowns/"
    client = Client(processes=False)
else:
    #Start cluster
    client = start(cpus=10)
    shps = glob.glob("/orange/ewhite/b.weinstein/NEON/draped/*.shp")
    CHMs = glob.glob("/orange/ewhite/NeonData/**/CanopyHeightModelGtif/*.tif",recursive=True)
    savedir = "/orange/idtrees-collab/"

#Get geoindex
geo_index = [re.search("(\d+_\d+)_image",x).group(1) for x in shps]
geo_index = np.unique(geo_index)

#year_futures = client.map(analysis.match_years, geo_index, shps=shps,savedir=savedir + "growth/")
falls_futures = client.map(analysis.tree_falls, geo_index, shps=shps, CHMs=CHMs, savedir=savedir + "treefall/")

#year_results = []
#for future in as_completed(year_futures):
    #try:
        #result = future.result()
        #results.append(result)
    #except Exception as e:
        #print("Year futures: {} failed with {}".format(future, e))   

fall_results = []
for future in as_completed(falls_futures):
    try:
        result = future.result()
        results.append(result)
    except Exception as e:
        print("Year futures: {} failed with {}".format(future, e))   

#wait(year_futures)    
wait(falls_futures)
