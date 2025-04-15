import os
import glob
import sys
import re
import json
import fiona
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pathlib
import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import shape, Polygon, mapping, Point, MultiPolygon, LinearRing
from shapely.validation import make_valid
from shapely.errors import GEOSException
from sentinelhub import SHConfig
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)
from utils.sentinel.parse_command_line import parse_command_line
from utils.generate_cadastral_id import generate_cadastral_id
#from utils.get_df import get_df

def close_polygon(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        exterior_coords = list(geom.exterior.coords)
        if exterior_coords[0] != exterior_coords[-1]:
            exterior_coords.append(exterior_coords[0])  # close the ring
        return Polygon(exterior_coords, holes=geom.interiors)
    return geom


def get_df(region, prov, cod_comune, comune):

  file_path = f"../ITALIA/{region}/{prov}/{cod_comune}_*/*_ple.gml"
  print(file_path)
  matching_files = glob.glob(file_path, recursive=False)
  print(f"matching_files: {matching_files}")
  fixed_geoms = []

  if matching_files:
    with fiona.open(matching_files[0]) as src:
      for feature in src:
        try:
          geom = shape(feature["geometry"])
          if not geom.is_valid:
              geom = geom.buffer(0)  # intenta reparar
          fixed_geoms.append(geom)
        except Exception as e:
          print(f"Error en feature {feature['id']}: {e}")

    # Ahora crea el GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=fixed_geoms, crs=src.crs)
    print(gdf)
  else:
    print('No matching file found')
    return pd.DataFrame()
  return gdf

def plot_image(image, factor=1, cmap = 'viridis',vmin=0, vmax=1):
  """
  Utility function for plotting RGB images.
  """
  plt.subplots(nrows=1, ncols=1, figsize=(15,7))
  print(vmin)
  print(vmax)

  plt.imshow(np.minimum(image*factor, vmax), cmap=cmap, vmin=vmin, vmax=vmax)
  plt.savefig("test.png")

def get_config(params,INSTANCE_ID = '3be8aaf7-7df4-4d22-b7af-f0d9dc4736c3'):
  
  with open(params['authentification']['path'], 'r') as file:
    inputs = yaml.safe_load(file)
  auth = inputs['authentification']
  if INSTANCE_ID:
    config = SHConfig()
    #config.instance_id = INSTANCE_ID
    config.sh_client_id = auth["sh_client_id"]
    config.sh_client_secret = auth["sh_client_secret"] 
    config.sh_base_url = auth["sh_base_url"] 
    config.sh_token_url = auth["sh_token_url"] 
    #config = SHConfig()
    #config.instance_id = INSTANCE_ID
  else:
    config = None
  return config
  
def all_bands_request(config, time_interval, region_bbox, region_size):
  evalscript_all_bands = """
  //VERSION=3
  function setup() {
      return {
          input: [{
              //bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"],
              bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"],
              units: "DN"
          }],
          output: {
              bands: 13,
              sampleType: "INT16"
          }
      };
  }

  function evaluatePixel(sample) {
      return [sample.B01,
              sample.B02,
              sample.B03,
              sample.B04,
              sample.B05,
              sample.B06,
              sample.B07,
              sample.B08,
              sample.B8A,
              sample.B09,
              //sample.B10,
              sample.B11,
              sample.B12];
  }
  """

  return SentinelHubRequest(
    evalscript=evalscript_all_bands,
    input_data=[
      SentinelHubRequest.input_data(
        #data_collection=DataCollection.SENTINEL2_L1C.define_from("s2l1c", service_url=config.sh_base_url),
        data_collection=DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=config.sh_base_url),
        time_interval=time_interval,
        mosaicking_order=MosaickingOrder.LEAST_CC,
      )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=region_bbox,
    size=region_size,
    config=config,
  )
def true_colors_request(config, time_interval, region_bbox, region_size):
  evalscript_true_color = """
  //VERSION=3
  function setup() {
      return {
          input: [{
              bands: ["B02", "B03", "B04"]
          }],
          output: {
              bands: 3
          }
      };
  }
  function evaluatePixel(sample) {
      return [sample.B04, sample.B03, sample.B02];
  }
  """
  return SentinelHubRequest(
    evalscript=evalscript_true_color,
    input_data=[
      SentinelHubRequest.input_data(
        #data_collection=DataCollection.SENTINEL2_L1C.define_from("s2l1c", service_url=config.sh_base_url),
        data_collection=DataCollection.SENTINEL2_L2A.define_from("s2l2a", service_url=config.sh_base_url),
        time_interval=time_interval,
        mosaicking_order=MosaickingOrder.LEAST_CC,
      )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=region_bbox,
    size=region_size,
    config=config,
  )

def custom_requeset(params, config, time_interval, region_bbox, region_size):
  #TODO: make sure that the collection, collection_alias, avalscript_custo, mosaiking_order and output_type are correct and exist as option before making the reques.
  #TODO: Add the parameters above to the yaml config file and add evalscript  to the Data directory as an option.
  request_params = params['request_params']
  with open(request_params['evalscript_path']) as file:
    evalscript_custom = file.read()
  
  collection = request_params['collection']
  collection_alias = request_params['collection_alias']
  other_args = request_params['other_args']
  if other_args == 'None':
    other_args = None
  print(type(other_args))
  if  request_params['mosaicking_order'] == 'None':
    mosaicking_order = None
  else:
    mosaicking_order = getattr(MosaickingOrder, request_params['mosaicking_order'])
  res = request_params['responses']
  responses = [SentinelHubRequest.output_response(key, getattr(MimeType, value)) for key, value in res.items()]


  '''
  responses=[
    SentinelHubRequest.output_response("default", getattr(MimeType, output_type)),
    SentinelHubRequest.output_response("userdata", MimeType.JSON),
    ],
  '''
  return SentinelHubRequest(
    evalscript=evalscript_custom,
    input_data=[
      SentinelHubRequest.input_data(
        data_collection=getattr(DataCollection, collection).define_from(collection_alias, service_url=config.sh_base_url),
        time_interval=time_interval,
        other_args = other_args,
        #mosaicking_order=getattr(MosaickingOrder, mosaicking_order),
        mosaicking_order= mosaicking_order,
      )
    ],
    responses = responses,
    bbox=region_bbox,
    size=region_size,
    config=config,
  )
def make_request(params,region_coords=(14.7434, 40.8638, 15.1123, 41.0615), output_name='test_region', crs=CRS.WGS84):
  config = get_config(params)
  resolution = params['request_params']['resolution']
  region_bbox = BBox(bbox=region_coords, crs=crs)
  region_size = bbox_to_dimensions(region_bbox, resolution=resolution)

  print("##################################")
  print(f"region_size: {region_size}")
  print("##################################")
  
  print(f"Image shape at {resolution} m resolution: {region_size} pixels")
  #print(list(MimeType)) # This is to list all the somple formats available
  '''
  slots = [('2018-12-01', '2019-02-28'), ('2019-03-01', '2019-04-30'), ('2019-05-01', '2019-06-30'), ('2019-07-01', '2019-08-31'),
           ('2019-12-01', '2020-02-28'), ('2020-03-01', '2020-04-30'), ('2020-05-01', '2020-06-30'), ('2020-07-01', '2020-08-31'),
           ('2020-12-01', '2021-02-28'), ('2021-03-01', '2021-04-30'), ('2021-05-01', '2021-06-30'), ('2021-07-01', '2021-08-31'),
           ('2021-12-01', '2022-02-28'), ('2022-03-01', '2022-04-30'), ('2022-05-01', '2022-06-30'), ('2022-07-01', '2022-08-31')]
  '''
  slots = params['request_params']['slots']
  if slots == None:
    start = datetime.datetime(2019, 1, 1)
    end = datetime.datetime(2022, 12, 31)
    n_chunks = 49
    #n_chunks = 2
    tdelta = (end - start) / n_chunks
    edges = [(start + i * tdelta).date().isoformat() for i in range(n_chunks)]
    slots = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
  
  print("Time windows:\n")
  print(slots)
  
  # create a list of requests
  #list_of_requests = [true_colors_request(config, slot, region_bbox, region_size) for slot in slots]
  #list_of_requests = [all_bands_request(config, slot, region_bbox, region_size) for slot in slots]
  list_of_requests = [custom_requeset(params, config, slot, region_bbox, region_size) for slot in slots]
  list_of_requests = [request.download_list[0] for request in list_of_requests]
  

  for i, l in enumerate(list_of_requests):
    l.save_response =True
    l.data_folder= params['request_params']['output_directory']
    l.save_response = True
    l.filename = f"{output_name}_{slots[i][0]}_{slots[i][1]}.zip"
  
  # download data with multiple threads
  data = SentinelHubDownloadClient(config=config).download(list_of_requests, max_threads=5, show_progress=True)

  
  # some stuff for pretty plots
  ncols = 12 
  nrows = 4
  aspect_ratio = region_size[0] / region_size[1]
  subplot_kw = {"xticks": [], "yticks": [], "frame_on": False}
  
  fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols * aspect_ratio, 5 * nrows), subplot_kw=subplot_kw)
  
  print(type(data))
  print(type(data[0]))
  for idx, data_i in enumerate(data):
    print(data[idx]['userdata.json'])
    image = data_i['default.tif']
    #print(image.shape)
    image = image#[:,:,[0, 0, 0]]
    ax = axs[idx // ncols][idx % ncols]
    ax.imshow(np.clip(image * 3.5 / 10000, 0, 1))
    ax.set_title(f"{slots[idx][0]}  -  {slots[idx][1]}", fontsize=10)
  
  plt.tight_layout()
  plt.savefig(f'samples/sentinel/all_{output_name}.png')
  exit()

def main(argv=None):
  if argv == None:
    argv = sys.argv[1:]
  args = parse_command_line(argv)
  crs = CRS.WGS84
  region_coords = (14.7434, 40.8638, 15.1123, 41.0615) # This our whole region of interest. Included AV and BN sections
  dataset_comune = {}
  isOnlyParcelsOfInterest = True

  #config_file = 'configs/default_parameters_sentinel.yaml'
  config_file = 'configs/parameters_sentinel_1.yaml'
  with open(config_file, 'r') as file:
    inputs = yaml.safe_load(file)
  params = inputs['params']


  input_files = [f for f in pathlib.Path().glob("../GEOJSON/FEUDI/GEOJSON_FEUDI/*.geojson")]
  for in_file in input_files:
    print(in_file)
    # Extract filename without path
    filename = str(in_file).rsplit("/", 1)[-1]  # Get last segment after the last "/"
    # Remove extension
    result = filename.rsplit(".", 1)[0]
    region, prov, cod_comune, comune = result.split('.')
    output_name = f'{region}_{prov}_{comune}'

    dataset_comune[f'{region}_{prov}_{comune}'] = None


    if isOnlyParcelsOfInterest:
      input_data = gpd.read_file(in_file, layer=comune)
      for fog, par, pol in zip(input_data.Foglio, input_data.Particella, input_data.geometry):
        if dataset_comune[f'{region}_{prov}_{comune}'] is None:
          dataset_comune[f'{region}_{prov}_{comune}'] = pol
        else:
          dataset_comune[f'{region}_{prov}_{comune}'] = make_valid(dataset_comune[f'{region}_{prov}_{comune}'].union(make_valid(pol)))

      if dataset_comune[f'{region}_{prov}_{comune}'] == None:
        print(f'The polygon for {region}_{prov}_{comune} is empty')
      else:
        print(f'Making the map request for {region}_{prov}_{comune}')
        print(dataset_comune[f'{region}_{prov}_{comune}'].bounds)
        region_coords = dataset_comune[f'{region}_{prov}_{comune}'].bounds
        make_request(params, region_coords, output_name, crs)
    else:
      if os.path.isfile(f"data/sentinel/{region}_{prov}_{comune}.json"):
        with open(f"data/sentinel/{region}_{prov}_{comune}.json") as jf:
          region_bbox = json.load(jf)
        region_coords = region_bbox['bbox']
        print(region_coords)
        make_request(params, region_coords, output_name, crs)
      else:
        comune_pd = get_df(region,prov, cod_comune, comune)
        print(f"data/sentinel/{region}_{prov}_{comune}.json")
        print(comune_pd.columns.tolist())
        print(comune_pd.head)
        for pol in comune_pd.geometry:
          if dataset_comune[f'{region}_{prov}_{comune}'] is None:
            dataset_comune[f'{region}_{prov}_{comune}'] = pol
          else:
            dataset_comune[f'{region}_{prov}_{comune}'] = make_valid(dataset_comune[f'{region}_{prov}_{comune}'].union(make_valid(pol)))

        if dataset_comune[f'{region}_{prov}_{comune}'] == None:
          print(f'The polygon for {region}_{prov}_{comune} is empty')
        else:
          print(f'Making the map request for {region}_{prov}_{comune}')
          print(dataset_comune[f'{region}_{prov}_{comune}'].bounds)
          region_coords = dataset_comune[f'{region}_{prov}_{comune}'].bounds
          make_request(params, region_coords, output_name, crs)
          out_bbox = {"bbox": region_coords}
          with open(f"data/sentinel/{region}_{prov}_{comune}.json", 'w') as fp:
            json.dump(out_bbox, fp)

    
    '''
    if fog is None:
    cadastral_id = generate_cadastral_id(cod_comune, int(fog), int(par))
    if not comune_pd.empty:
      polygon = comune_pd[comune_pd.gml_id == cadastral_id].geometry
      for pol in polygon:
        if dataset_comune[f'{region}_{prov}_{comune}'] is None:
          dataset_comune[f'{region}_{prov}_{comune}'] = pol
        else:
          dataset_comune[f'{region}_{prov}_{comune}'] = dataset_comune[f'{region}_{prov}_{comune}'].union(pol)
    '''



if __name__== '__main__':
  status = main()
  sys.exit(status)
