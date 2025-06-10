DEBUG = False
import sys
import yaml
import argparse
import re
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import shape

from utils.parse_command_line import parse_command_line
from utils.get_df import get_df
from utils.generate_cadastral_id import generate_cadastral_id


def transform_string(s):
  s_out = re.sub(r'[^A-Za-z0-9_ ]', '', s).upper()
  s_out = re.sub(r'[ ]', '_', s_out)
  return s_out 

def main(argv=None):
  if argv == None:
    argv = sys.argv[1:]
  args = parse_command_line(argv,is_local=True)

  config_file = args['config_file']
  with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
  params = config['params']
  parcels_file = params['requests_file']['path']
  print(f"Looking for the parcels listed in  {parcels_file}")
  requests_pd = pd.read_excel(parcels_file, 
                             index_col=0,
                             dtype = params['requests_file']['format'])
  provinces = params['requests_file']['provinces']
  out_file_list = []
  fail_list = []
  for region in provinces.keys():
    for prov in provinces[region]:
      req_prov_pd = requests_pd.query(f'Regione=="{region}" and SiglaProvincia == "{prov}"')
      req_com_pd = req_prov_pd[~req_prov_pd["CodComune"].duplicated()][['CodComune', 'Comune']]
      for i, c in req_com_pd.iterrows():
        comune_pd = get_df(region,prov, c.CodComune, c.Comune)
        #print(f'{region}-{prov}-Retrieving images for the "comune": {c.Comune}')
        feature_collection = {}
        feature_collection['type'] = "FeatureCollection"
        feature_collection['name'] = f"{region}_{prov}_{c.Comune}"
        #feature_collection['crs'] = { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } }
        features = [] 

        if comune_pd.empty:
          print("here")
          continue
        for i, (foglio, particella) in enumerate(zip(req_prov_pd[req_prov_pd.CodComune == c.CodComune].Foglio, req_prov_pd[req_prov_pd.CodComune == c.CodComune].Particella)):
          #print(f"Foglio {foglio}. Retrieving parcel {particella}")
          feature = {}
          feature['id'] = i+1
          feature['type'] = 'Feature'
          feature['properties'] = {"id": i+1,
                  "Foglio": foglio, 
                  "Particella": particella}
          feature['geometry']={}

          cadastral_id = generate_cadastral_id(c.CodComune, foglio, particella)
          image_name = f"{region}_{prov}_{transform_string(c.Comune)}_{foglio}_{particella}"
          polygon = comune_pd[comune_pd.gml_id == cadastral_id].geometry
          if len(polygon.to_numpy()) > 0:
            poly = polygon.to_numpy()[0] 
            feature['geometry']['type'] = poly.geom_type
            coordinates = []

            if poly.geom_type != 'Polygon':
              for pol in poly.geoms:
                for p in pol.interiors:
                  coordinates.append([ list(coord) for coord in p.coords])
                coordinates.append([ list(coord) for coord in pol.exterior.coords])
              coordinates = [coordinates]
            else:
              coordinates.append([ list(coord) for coord in poly.exterior.coords])
            feature['geometry']['coordinates'] = coordinates 
            features.append(feature)
          else:
            fail_list.append(image_name)
        feature_collection['features'] = features
        geometries = [shape(feat["geometry"]) for feat in features]
        properties = [feat["properties"] for feat in features]
        ids = [feat["id"] for feat in features]
        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
        gdf["id"] = ids
        if DEBUG:
          print(gdf.to_json())
        print('here1')
        out_file_name = f"data/geojsons/{region}_{prov}_{transform_string(c.Comune)}.geojson"
        gdf.to_file(out_file_name, driver='GeoJSON')
        out_file_list.append(out_file_name)
        print('here2')
  out_file = open(f"data/geojsons/file_list.txt", 'w')
  for p in out_file_list:
    print(p)
    out_file.write(p+"\n")
  out_file.close()
  fail_file = open(f"data/geojsons/fail_list.txt", 'w')
  for fail in fail_list:
    print(fail)
    fail_file.write(fail+"\n")
  fail_file.close()

if __name__=="__main__":
  status = main()
  sys.exit(status)
