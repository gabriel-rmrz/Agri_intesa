import pandas as pd
import geopandas as gpd
from shapely.errors import GEOSException

def get_df(region, prov, cod_comune, comune):
  file_path = f"../ITALIA/{region}/{prov}/{cod_comune}_{comune}/{cod_comune}_{comune}_ple.gml"
  try:
    gdf = gpd.read_file(file_path)
  except GEOSException as e:
    print(f"Warning: {e}")
    return pd.DataFrame()
  return gdf
