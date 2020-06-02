#test sampling
import os
import sys
import glob
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.getcwd()))

from analysis import sampling
from analysis.check_site import get_site, get_year

def test_simulate_plot():
    #Find files
    tile_list = glob.glob("../data/*.shp")
    
    #Get site names
    df = pd.DataFrame({"path":tile_list})
    df["year"] = df.path.apply(lambda x: get_year(x))
    df["site"] = df.path.apply(lambda x: get_site(x))
    
    #Construct list of site+year combinations
    site_lists = df.groupby(['site','year']).path.apply(list).to_dict()     

    for x in site_lists:
        for i in np.arange(2):
            df = sampling.run(site_lists[x])
            print(df)
            assert df.shape == (1,5)