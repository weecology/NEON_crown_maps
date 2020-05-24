#test submission document
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from .. import create_submission

def test_submission():
    eval_path = "/Users/ben/Documents/NeonTreeEvaluation/evaluation/RGB/benchmark_annotations.csv"
    CHM_dir = "/Users/ben/Documents/NeonTreeEvaluation/evaluation/CHM/"
    boxes = create_submission.submission(eval_path,CHM_dir,iou_threshold=0.15,saved_model="/Users/ben/Downloads/finetuned_weights_20200512_133157.h5")
    assert all(boxes.columns == ["plot_name","xmin","ymin","xmax","ymax","score","label"])
    boxes.to_csv("output/20200512_133157_filtered.csv")
    
#def test_submission_no_chm():
    #eval_path = "/Users/ben/Documents/NeonTreeEvaluation/evaluation/RGB/benchmark_annotations.csv"
    #CHM_dir = "/Users/ben/Documents/NeonTreeEvaluation/evaluation/CHM/"
    #boxes = create_submission.submission_no_chm(eval_path,CHM_dir)
    #assert all(boxes.columns == ["plot_name","xmin","ymin","xmax","ymax","score","label"])
    #boxes.to_csv("output/CurrentModel.csv")