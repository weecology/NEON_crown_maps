"""Predict all plots which have NEON field data"""
from deepforest import deepforest
import os
from predict import project
from LIDAR import non_zero_99_quantile
import rasterstats
from predict import predict_tiles
import geopandas as gp
import pandas as pd

def run(eval_path, CHM_dir, min_height=3):
    
    #Predict
    model = deepforest.deepforest()
    model.use_release()
    
    #Load field data
    field_data = pd.read_csv("Figures/vst_field_data.csv")
    
    #Load field data locations
    site_shp = gp.read_file("Figures/All_NEON_TOS_Plots_V7/All_NEON_TOS_Plot_Polygons_V7.shp")

    #Which locations have data
    site_shp = site_shp[site_shp.plotID.isin(field_data.plotID.unique())]
    
    site_shp["path"] = site_shp.plotID.apply(lambda x: "{}.tif".format(os.path.join(eval_path,x)))
    
    #Predict each unique tile
    tiles_to_predict = site_shp["path"].unique()
    results = [ ]
    for tile in tiles_to_predict:
        try:
            result = model.predict_tile(tile,return_plot=False,patch_size=400, iou_threshold=0.1)
            result["plot_name"] = os.path.splitext(os.path.basename(tile))[0]
            results.append(result)
        except Exception as e:
            print(e)
            continue
    
    boxes = pd.concat(results)
    
    #Create plot name groups
    boxes_grouped = boxes.groupby('plot_name')    
    plot_groups = [boxes_grouped.get_group(x) for x in boxes_grouped.groups]
    
    #Set RGB dir
    rgb_dir = os.path.dirname(eval_path)
    
    #Project
    threshold_boxes = []
    for x in plot_groups:
        
        plot_name = x.plot_name.unique()[0]
        #Look up RGB image for projection
        image_path = "{}/{}.tif".format(rgb_dir,plot_name)
        result = project(image_path,x)
        
        #Extract heights
        chm_path = "{}_CHM.tif".format(os.path.join(CHM_dir, plot_name))
        
        try:
            height_dict = rasterstats.zonal_stats(result, chm_path, stats="mean", add_stats={'q99':non_zero_99_quantile})
        except Exception as e:
            print("{} raises {}".format(plot_name,e))
            continue
            
        x["height"]  = [g["q99"] for g in height_dict]
        
        #Merge back to the original frames
        threshold_boxes.append(x)
    
    threshold_boxes = pd.concat(threshold_boxes)
    threshold_boxes = threshold_boxes[threshold_boxes.height > min_height]
    threshold_boxes["area"] = (threshold_boxes["top"] - threshold_boxes["bottom"]) * (threshold_boxes["right"] - threshold_boxes["left"])   
    threshold_boxes = threshold_boxes[["plot_name","xmin","ymin","xmax","ymax","score","label","height","area"]]
    
    return threshold_boxes

def create_shapefiles(eval_path, CHM_dir, min_height=3, save_dir="."):
    
    #Predict
    model.config[""]
    model = deepforest.deepforest()
    model.use_release()
    
    #Load field data
    field_data = pd.read_csv("Figures/vst_field_data.csv")
    
    #Load field data locations
    site_shp = gp.read_file("Figures/All_NEON_TOS_Plots_V7/All_NEON_TOS_Plot_Polygons_V7.shp")

    #Which locations have data
    site_shp = site_shp[site_shp.plotID.isin(field_data.plotID.unique())]
    
    site_shp["path"] = site_shp.plotID.apply(lambda x: "{}.tif".format(os.path.join(eval_path,x)))
    
    #Predict each unique tile
    tiles_to_predict = site_shp["path"].unique()
    for tile in tiles_to_predict:
        try:
            result = model.predict_tile(tile,return_plot=False,patch_size=400,iou_threshold=0.1)
            result["plot_name"] = os.path.splitext(os.path.basename(tile))[0]
        except Exception as e:
            print(e)
            continue
        
        if result.empty:
            continue
        
        #Set RGB dir
        rgb_dir = os.path.dirname(eval_path)
        plot_name = result.plot_name.unique()[0]
        
        #Look up RGB image for projection
        image_path = "{}/{}.tif".format(rgb_dir,plot_name)
        projected_boxes = project(image_path,result)
        
        #Extract heights
        chm_path = "{}_CHM.tif".format(os.path.join(CHM_dir, plot_name))
        
        try:
            height_dict = rasterstats.zonal_stats(projected_boxes, chm_path, stats="mean", add_stats={'q99':non_zero_99_quantile})
        except Exception as e:
            print("{} raises {}".format(plot_name,e))
            continue
            
        #Limit by height
        projected_boxes["height"]  = [g["q99"] for g in height_dict]
        projected_boxes = projected_boxes[projected_boxes["height"]>3]  
        
        if not projected_boxes.empty:
            #Write
            shp_path = "{}.shp".format(os.path.join(save_dir,plot_name))
            projected_boxes.to_file(shp_path, driver='ESRI Shapefile')   
            print("{} written".format(shp_path))
                

if __name__=="__main__":
    
    df = run( eval_path="/Users/ben/Documents/NeonTreeEvaluation/evaluation/RGB/", CHM_dir="/Users/ben/Documents/NeonTreeEvaluation/evaluation/CHM/")
    df.to_csv("Figures/plot_predictions.csv")
    
    #df = create_shapefiles(eval_path="/Users/ben/Documents/NeonTreeEvaluation/evaluation/RGB/", CHM_dir="/Users/ben/Documents/NeonTreeEvaluation/evaluation/CHM/",save_dir="/Users/Ben/Dropbox/Weecology/Crowns/Site_Predictions/")
                    