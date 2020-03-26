#Check tile utility functions. Remove poor quality tiles from pool
from PIL import Image
import rasterio
import numpy as np


#Tile checks for validity

def check_RGB(tile_path):
    """Return name if passes checks, None if not"""
    #Load image    
    try:
        raster = Image.open(tile_path)
    except:
        print("Image {} is corrupt".format(tile_path))
        return None
    
    #Test if a large portion of the image is black
    numpy_image = np.array(raster)    
    is_black =np.sum(np.all(numpy_image == [0,0,0], axis=-1))/numpy_image.size
    
    #If more than 3% black, remove edge tile
    if is_black > 0.03:
        print("{} is an edge tile, {number:.{digits}f}% black pixels".format(tile_path,number=is_black*100,digits=1))
        return None
    
    return tile_path

#Check corresponding CHM for validity
def check_CHM(tile_path):
    #Load
    CHM = rasterio.open(tile_path)
    numpy_array = CHM.read()
    
    #Proportion NoDATA and Empty (np.nan)
    proportion_empty =np.sum(np.isnan(numpy_array))/numpy_array.size    
    
    proportion_nodata=np.sum(numpy_array==CHM.nodatavals)/numpy_array.size
    
    if proportion_nodata + proportion_empty > 0.2:
        return False
    else:
        return True

def get_year(path):
    basename = os.path.basename(path)
    year = re.search("^(\d+)_",basename).group(1)
    return year

def get_site(path):
    basename = os.path.basename(path)                
    site = re.search("^\d+_(\w+)_\d_",basename).group(1)
    return site