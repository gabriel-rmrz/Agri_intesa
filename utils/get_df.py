DEBUG= False
import glob
import fiona
import os
import pandas as pd
import geopandas as gpd
from shapely.errors import GEOSException
from shapely.geometry import shape, Polygon, Point, MultiPolygon


def get_df(region, prov, cod_comune, comune):

  '''
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
  '''

  file_path = f"../ITALIA/{region}/{prov}/{cod_comune}_*/*_ple.gml"
  matching_files = glob.glob(file_path, recursive=False)
  if DEBUG:
    print(file_path)
    print(f"matching_files: {matching_files}")
  fixed_geoms = []
  gml_id = []


  if matching_files:
    with fiona.open(matching_files[0]) as src:
      for feature in src:
        try:
          geom = shape(feature["geometry"])
          if not geom.is_valid:
              geom = geom.buffer(0)  # intenta reparar
          fixed_geoms.append(geom)
          gml_id.append(feature["properties"]["gml_id"])
        except Exception as e:
          print(f"Error en feature {feature['id']}: {e}")

    # Ahora crea el GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=fixed_geoms, crs=src.crs)
    gdf["gml_id"] = gml_id
    if DEBUG:
      print(gdf)
  else:
    print('No matching file found')
    return pd.DataFrame()
  return gdf
