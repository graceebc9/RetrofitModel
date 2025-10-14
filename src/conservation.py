import geopandas as gpd 

def load_conservation_shapefile(path = '/Users/gracecolverd/Downloads/Conservation_Areas_-5503574965118299320/Conservation_Areas.shp'):
    
    cons = gpd.read_file(path)
    return cons 