import geopandas as gp
import rasterio
import os


def shapefile_to_annotations(shapefile, rgb, savedir="."):
    """
    Convert a shapefile of annotations into annotations csv file for DeepForest training and evaluation
    Args:
        shapefile: Path to a shapefile on disk. If a label column is present, it will be used, else all labels are assumed to be "Tree"
        rgb: Path to the RGB image on disk
        savedir: Directory to save csv files
    Returns:
        None: a csv file is written
    """
    try:
        import geopandas
    except:
        raise EnvironmentError(
            "Geopandas not installed by default, as it is only used for this utility. See geopandas installation: http://geopandas.org/install.html "
        )

    #Read shapefile
    gdf = gp.read_file(shapefile)

    #get coordinates
    df = gdf.geometry.bounds

    #raster bounds
    with rasterio.open(rgb) as src:
        left, bottom, right, top = src.bounds

    #Transform project coordinates to image coordinates
    df["tile_xmin"] = df.minx - left
    df["tile_xmin"] = df["tile_xmin"].astype(int)

    df["tile_xmax"] = df.maxx - left
    df["tile_xmax"] = df["tile_xmax"].astype(int)

    #UTM is given from the top, but origin of an image is top left

    df["tile_ymax"] = top - df.miny
    df["tile_ymax"] = df["tile_ymax"].astype(int)

    df["tile_ymin"] = top - df.maxy
    df["tile_ymin"] = df["tile_ymin"].astype(int)

    #Add labels is they exist
    if "label" in gdf.columns:
        df["label"] = gdf["label"]
    else:
        df["label"] = "Tree"

    #add filename
    df["image_path"] = rgb

    #select columns
    result = df[[
        "image_path", "tile_xmin", "tile_ymin", "tile_xmax", "tile_ymax", "label"
    ]]
    result = result.rename(columns={
        "tile_xmin": "xmin",
        "tile_ymin": "ymin",
        "tile_xmax": "xmax",
        "tile_ymax": "ymax"
    })
    image_name = os.path.splitext(os.path.basename(rgb))[0]
    csv_filename = os.path.join(savedir, "{}.csv".format(image_name))

    #write file
    result.to_csv(csv_filename, index=False)


if __name__ == "__main__":
    shapefile = "/Users/ben/Downloads/predictions/2018_BART_4_318000_4876000_image.shp"
    rgb = "/Users/ben/Downloads/predictions/2018_BART_4_318000_4876000_image.tif"

    shapefile_to_annotations(shapefile, rgb)
