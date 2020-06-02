#test rasterize
import os
from analysis import rasterize

def test_run():
    path = "/Users/ben/Dropbox/Weecology/Crowns/examples/2019_BART_5_320000_4881000_image.shp"
    rasterize.run(path, rgb_dir="/Users/ben/Dropbox/Weecology/Crowns/examples/", savedir="output/")
    output="output/{}_rasterized.tif".format(os.path.splitext(os.path.basename(path))[0])
    assert os.path.exists(output)