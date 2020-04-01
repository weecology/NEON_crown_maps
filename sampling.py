#Sampling module. Simulate 40m NEON plots across the landscape.
import glob
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Point
from start_cluster import start
from check_site import get_site, get_year
import dask

def select_tile(tile_list):
    """
    Args:
        tile_list: a list of tiles to choose from
    Returns:
        shp: a path to a shapefile
    """
    import random
    
    #Select one tile
    shp = random.choice(tile_list)
    return shp

def create_plot(gdf):
    """Select a point to serve as plot center
    gdf: A geodataframe of predictions
    """
    
    import random 
    
    #Find spatial extent
    tile_left, tile_bottom, tile_right, tile_top  = gdf.total_bounds
    
    #Select a point
    p = Point(random.uniform(tile_left, tile_right), random.uniform(tile_bottom, tile_top))    
    plot_center_x, plot_center_y = list(p.coords)[0]
    
    return plot_center_x, plot_center_y

def create_subplots(plot_center_x,plot_center_y,length = 40, n=2):
    """
    length: size in m of one side of the plot
    n: number of subplots quarters
    """
    import random
    
    #Get plot edges
    plot_left = plot_center_x - length/2
    plot_bottom = plot_center_y - length/2
    plot_right = plot_center_x + length/2
    plot_top = plot_center_y + length/2 
    
    plot_bounds = box(plot_left,plot_bottom,plot_right, plot_top)
    
    #Select subplots
    subplot_list = random.sample([1,2,3,4], n)
    subplot_bounds = [ ]
    
    #For each subplot get quarter coordinates
    for subplot in subplot_list:
        if subplot == 1:
            selected_subplot = box(plot_left, plot_center_y, plot_center_x, plot_top)
        elif subplot == 2:
            selected_subplot = box(plot_center_x, plot_center_y, plot_right, plot_top)
        elif subplot ==3:
            selected_subplot = box(plot_right, plot_bottom, plot_center_x, plot_center_y)
        elif subplot == 4:
            selected_subplot = box(plot_center_x, plot_bottom, plot_right, plot_center_y)
        subplot_bounds.append(selected_subplot)    
        
        return subplot_bounds
    
#create a simulated plot
def simulate_plot(shp):
    """
    Simulate a 40mx40m plot with 2 subplots for tree statistics
    """
    #Read shapefile
    df = gpd.read_file(shp)
    
    #select plot center
    plot_center_x, plot_center_y = create_plot(df)
    subplot_bounds = create_subplots(plot_center_x, plot_center_y)
    
    #Two subplots within the plot
    plot_data = [ ]    
    for subplot in subplot_bounds:
        selected_trees = select_trees(df, subplot)
        plot_data.append(selected_trees)
   
    plot_data =  pd.concat(plot_data)
    
    #Calculate statistics
    tree_density = calculate_density(plot_data)
    average_height = calculate_height(plot_data)
    
    #Create data holder
    data =  {
        "path": [shp],
        "plot_center_x": plot_center_x,
        "plot_center_y": plot_center_y,
        "tree_density": [tree_density],
        "average_height": [average_height]
    }
    
    return pd.DataFrame(data)
        
def select_trees(gdf, subplot):
    """
    Args:
        gdf: a geopandas dataframe
        subplot: a shapely box
    Returns:
        selected_trees: pandas dataframe of trees
    """
    selected_trees = gdf[gdf.intersects(subplot)]
    return selected_trees

#Calculate tree density
def calculate_density(plot_data):
    return plot_data.shape[0]

#Calculate mean height
def calculate_height(plot_data):
    return plot_data.height.mean()

def run(tile_list):
    
    #Load random at runtime to set state
    import random
    
    #Select tile
    shp = select_tile(tile_list)
    
    #Create a plot
    results = simulate_plot(shp)
    
    return results
    
if __name__ == "__main__":

    #Start dask client
    client = start(cpus=80, mem_size="3GB")
    
    #Get pool of predictions    
    shps = glob.glob("/orange/ewhite/b.weinstein/NEON/draped/*.shp")
    
    #Get site names
    df = pd.DataFrame({"path":shps})
    df["year"] = df.path.apply(lambda x: get_year(x))
    df["site"] = df.path.apply(lambda x: get_site(x))
    
    #Construct list of site+year combinations
    site_lists = df.groupby(['site','year']).path.apply(list).to_dict()
    
    tile_lists = [ ]
    for key, value in site_lists.items():
        tile_lists.append(value)
    
    simulation_results = [ ]
    for x in tile_lists:
        for i in np.arange(10000):
            result = dask.delayed(run)(x)
            simulation_results.append(result)
    results = dask.compute(*simulation_results)
    
    #Combine results
    results = pd.concat(results)
    results["year"] = results.path.apply(lambda x: get_year(x))
    results["site"] = results.path.apply(lambda x: get_site(x))
    
    results.to_csv("Figures/sampling.csv", index=False)
        
    
        