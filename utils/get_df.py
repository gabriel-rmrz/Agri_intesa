import glob
import os
import pandas as pd
import geopandas as gpd
from shapely.errors import GEOSException


def get_df(region, prov, cod_comune, comune):

  file_path = f"../ITALIA/{region}/{prov}/{cod_comune}_*/*_ple.gml"
  matching_files = glob.glob(file_path, recursive=False)
  if matching_files:
    try:
      gdf = gpd.read_file(matching_files[0])
    except GEOSException as e:
      print(f"Warning: {e}")
      return pd.DataFrame()
  else:
    print('No matching file found')
    return pd.DataFrame()

  return gdf
