#Test Hard Negative Mining
from .. import hard_mining

def test_run():
    prediction_path = "/Users/ben/Downloads/predictions/2018_BART_4_318000_4876000_image.shp"
    image_path = "/Users/ben/Downloads/predictions/2018_BART_4_318000_4876000_image.tif"
    hard_mining.run(prediction_path = prediction_path, image_path=image_path)