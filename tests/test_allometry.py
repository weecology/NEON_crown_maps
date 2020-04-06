#Test allometry
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import glob
import pandas as pd
from .. import allometry
from ..check_site import get_site, get_year

def test_run():
    shps = glob.glob("data/*.shp")    
    
    #Get site names
    df = pd.DataFrame({"path":shps})
    df["year"] = df.path.apply(lambda x: get_year(x))
    df["site"] = df.path.apply(lambda x: get_site(x))
    
    #Construct list of site+year combinations
    site_lists = df.groupby(['site','year']).path.apply(list).to_dict()
    
    #For each site year
    model_fit = {}
    for key, value in site_lists.items():
        model_fit[key] = allometry.run(value)
    
    modeldf = pd.DataFrame(model_fit)
    modeldf.to_csv("output/allometry.csv")
        
    

