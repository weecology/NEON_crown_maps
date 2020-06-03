from deepforest import deepforest
import os
from crown_maps.predict import project
from crown_maps.LIDAR import non_zero_99_quantile
import rasterstats
import pandas as pd

def submission_no_chm(eval_path, CHM_dir, min_height=3):
    
    #Predict
    model = deepforest.deepforest()
    model.use_release()
    boxes = model.predict_generator(eval_path)
    
    boxes = boxes[["plot_name","xmin","ymin","xmax","ymax","score","label"]]
    
    return boxes


def submission(eval_path, CHM_dir, min_height=3,iou_threshold=0.15, saved_model=None, tiles_to_predict=None):
    
    #Predict
    if saved_model:
        model = deepforest.deepforest(saved_model=saved_model)
    else:
        model = deepforest.deepforest()
        model.use_release()
    
    if not tiles_to_predict:
        df = pd.read_csv(eval_path,names=["plot_name","xmin","ymin","xmax","ymax","label"])
        tiles_to_predict = df.plot_name.unique()
        tiles_to_predict = [os.path.join(os.path.dirname(eval_path),x) for x in tiles_to_predict]
    
    results = []
    for tile in tiles_to_predict:
        try:
            result = model.predict_tile(tile,return_plot=False,patch_size=400, iou_threshold=iou_threshold)
            result["plot_name"] = os.path.splitext(os.path.basename(tile))[0]
            results.append(result)
        except Exception as e:
            print(e)
            continue    
        
    #Create plot name groups
    boxes = pd.concat(results)
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
    
    threshold_boxes = threshold_boxes[["plot_name","xmin","ymin","xmax","ymax","score","label"]]
    
    return threshold_boxes

if __name__=="__main__":
    
    #Just xml files
    #submission(
        #eval_path="/home/b.weinstein/NeonTreeEvaluation/evaluation/RGB/benchmark_annotations.csv",
        #CHM_dir="/home/b.weinstein/NeonTreeEvaluation/evaluation/CHM/"
    #)
    tiles_to_predict = glob.glob("/home/b.weinstein/NeonTreeEvaluation/evaluation/RGB/*.tif") 
    df = submission(tiles_to_predict=tiles_to_predict, CHM_dir="/home/b.weinstein/NeonTreeEvaluation/evaluation/CHM/")    
    df.to_csv("../Figures/all_images_submission.csv")