#Sampling module. Simulate 40m NEON plots across the landscape.
import glob
from shapely.geometry import box
import geopandas as gpd
import pandas as pd
from start_cluster import start

def select_tile(dirname):
    """
    Args:
        dirname: a directory to search for completed .shp predictions
    Returns:
        shp: a path to a shapefile
    """
    return shp

#create a simulated plot
def simulate_plot(shp):
    """
    Simulate a 40mx40m plot with 2 subplots for tree statistics
    """
    
    #Read shapefile
    df = gpd.read_file(shp)
    plot_data = [ ]
    
    #Two subplots within the plot
    for x in np.arange(2):
        box = create_subplot(shp)
        selected_trees = select_trees(df, box)
        plot_data.append(selected_trees)
   
    plot_data =  pd.concat(pd.DataFrame(plot_data))
    
    #Calculate statistics
    tree_density = calculate_density(plot_data)
    average_height = calculate_height(plot_data)
    
    #Create data holder
    data =  {
        "tile": shp,
        "left": box[0],
        "bottom": box[1],
        "right": box[2],
        "top": box[3],
        "tree_density":tree_density,
        "average_height": average_height
    }
    
    return data
        
def select_trees(box):
    """Return a pandas dataframe of trees based on shapely box"""
    pass

def create_subplot(box):
    return box

#Calculate tree density
def calculate_density(box):
    pass

#Calculate mean height
def calculate_height():
    pass

def run(dirname):
    #Select tile
    shp = select_tile(dirname)
    
    #Create a plot
    results = simulate_plot()
    return results
    
    
if __name__ == "__main__":

    #Start dask client
    client = start(cpus=80)

    #Get pool of predictions
    shps = glob.glob("/orange/ewhite/b.weinstein/NEON/draped/*.shp")
    
    simulation_futures = client.map(run, shps)
    results = [x.result() for x in simulation_futures]
    
    #Combine results
    df = pd.DataFrame(results)
    df.to_csv("/orange/idtrees-collab/sampling.csv")
    
    
        