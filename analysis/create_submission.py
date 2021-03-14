#srun -p gpu --gpus=1 --time 1:00:00 --mem 20GB --pty -u bash -i

#module load tensorflow/1.14.0

#export PATH=${PATH}:/home/b.weinstein/miniconda3/envs/crowns/bin/
#export PYTHONPATH=${PYTHONPATH}:/home/b.weinstein/miniconda3/envs/crowns/lib/python3.7/site-packages/
#export LD_LIBRARY_PATH=/home/b.weinstein/miniconda3/envs/crowns/lib/:${LD_LIBRARY_PATH}

from deepforest import deepforest
import os
from crown_maps.predict import project
from crown_maps.LIDAR import non_zero_99_quantile
import rasterstats
import pandas as pd
import glob

def submission_no_chm(tiles_to_predict, iou_threshold=0.15):
    #Predict
    results = []
    model = deepforest.deepforest()
    model.use_release()    
    for path in tiles_to_predict:   
        try:
            result = model.predict_image(path,return_plot=False)    
            result["plot_name"] = os.path.splitext(os.path.basename(path))[0]
            results.append(result)
        except Exception as e:
            print(e)
            continue    
        
    #Create plot name groups
    boxes = pd.concat(results)
    
    return boxes


def submission(CHM_dir=None, RGB_dir=None, min_height=3,iou_threshold=0.15, saved_model=None, tiles_to_predict=None):
    results = []
    model = deepforest.deepforest()
    model.use_release()    
    for path in tiles_to_predict:   
        try:
            result = model.predict_image(path,return_plot=False)    
            result["plot_name"] = os.path.splitext(os.path.basename(path))[0]
            results.append(result)
        except Exception as e:
            print(e)
            continue    
    
    boxes = pd.concat(results)
    boxes_grouped = boxes.groupby('plot_name')    
    plot_groups = [boxes_grouped.get_group(x) for x in boxes_grouped.groups]
        
    #Project
    threshold_boxes = []
    for x in plot_groups:
        plot_name = x.plot_name.unique()[0]
        print(plot_name)
        #Look up RGB image for projection
        image_path = "{}/{}.tif".format(RGB_dir,plot_name)
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
    
    #raw chm
    tiles_to_predict = glob.glob("/orange/idtrees-collab/NeonTreeEvaluation/evaluation/RGB/*.tif") 
    df = submission_no_chm(tiles_to_predict)
    df.to_csv("all_images_submission_NOCHM.csv")
    
    df = submission(tiles_to_predict=tiles_to_predict, RGB_dir = "/orange/idtrees-collab/NeonTreeEvaluation/evaluation/RGB/", CHM_dir="/orange/idtrees-collab/NeonTreeEvaluation/evaluation/CHM/")    
    df.to_csv("all_images_submission_CHM.csv")
