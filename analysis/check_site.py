#Site check
import glob
import os
import re
import pandas as pd
from distributed import Client, as_completed

from main import find_files, lookup_CHM_path
from crown_maps.start_cluster import start
from crown_maps import verify

def get_year(path):
                basename = os.path.basename(path)
                year = re.search("^(\d+)_",basename).group(1)
                return year
                
def get_site(path):
                basename = os.path.basename(path)                
                site = re.search("^\d+_(\w+)_\d_",basename).group(1)
                return site

if __name__  == "__main__":
                
                client = start(cpus = 50, mem_size ="5GB")
                #client = Client()
                
                lidar_pool = glob.glob("/orange/ewhite/NeonData/**/CanopyHeightModelGtif/*.tif",recursive=True)
                rgb_list = glob.glob("/orange/ewhite/NeonData/**/Mosaic/*image.tif",recursive=True)
                site_list = ["DELA"]
                year_list = ["2019","2018"]
                target_list=[]
                
                rgb=find_files(rgb_list,site_list,year_list)
                print("There are {} files found in dir".format(len(rgb)))
                
                RGB_verification = client.map(verify.check_RGB, rgb)
                rgb_verified = [x.result() for x in RGB_verification]
                
                #Remove None
                rgb_verified  = [x for x in rgb_verified if x]
                print("There are {} verified RGB tiles before checking LiDAR".format(len(rgb_verified)))
                
                CHM_verification = [ ]
                for x in rgb_verified:
                                try:
                                                lidar_path = lookup_CHM_path(x, lidar_pool, shp=False)
                                                if lidar_path:  
                                                                check = client.submit(verify.check_CHM,lidar_path)
                                                else:
                                                                print("{} has no matching CHM".format(x))
                                                                chm_path = None
                                                CHM_verification.append(check)
                                except Exception as e:                
                                                print("Path CHM {} lookup failed with {}".format(check,e))
                
                CHM_verified = cpu_client.gather(CHM_verification)
                print("There are {} verified CHM tiles before checking matches".format(len(CHM_verified)))
                
                final_rgb_list = [rgb_verified[index] for index, x in enumerate(CHM_verified) if x]
                print("There are {} RGB tiles with matching verified CHMs".format(len(final_rgb_list)))
                                
                df = pd.DataFrame({"path":final_rgb_list})
                             
                df["year"] = df.path.apply(lambda x: get_year(x))
                df["site"] = df.path.apply(lambda x: get_site(x))
                site_totals = df.groupby(["site","year"]).size()
                print(site_totals)
                
