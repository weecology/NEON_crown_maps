#Test Hard Negative Mining
from analysis import hard_mining

def test_run():
    prediction_path = "/Users/ben/Documents/NEON_crown_maps/tests/2019_SOAP_4_306000_4099000_image.shp"
    image_path = "/Users/ben/Documents/NEON_crown_maps/tests/2019_SOAP_4_306000_4099000_image.tif"
    hard_mining.run(prediction_path = prediction_path, image_path=image_path)