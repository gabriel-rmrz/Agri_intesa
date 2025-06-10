DEBUG=False
import os
import sys
import pathlib
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from shapely.geometry import shape, Polygon, mapping, Point, MultiPolygon, LinearRing
from shapely.validation import make_valid
from shapely.errors import GEOSException
from sentinelhub import SHConfig
from rasterio.mask import mask

from utils.sentinel.parse_command_line import parse_command_line
from utils.generate_cadastral_id import generate_cadastral_id
#from utils.get_df import get_df


def plot_1_band(im,dir_path):
  fig = plt.figure(figsize=(6,6))
  ax1 = fig.add_subplot(1,1,1)
  ax1.set_title("Sentinel 1")
  ax1.imshow(im)
  ax1.set_axis_off()
  fig.savefig(dir_path+".png")
  plt.close(fig)

def compute_parameters(masked_image,dir_path, dir_name, i_pol,  prefix=""):
  # bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12"]

  # Getting values of reflectance (Following Katherine's PDF)
  #r_im = masked_image/10000
  r_im = masked_image/10000
  # Coomputing NDVI = (NIR-RED)/(NIR+RED) # NIR: Band 8
  # RED: Band 4
  ndvi_num = (r_im[7]-r_im[3])
  ndvi_den = r_im[7]+r_im[3]
  ndvi = np.divide(ndvi_num, ndvi_den, out= np.zeros_like(ndvi_num), where=ndvi_den!=0)
  plot_1_band(ndvi, f"{dir_path}/ndvi/{prefix}/{dir_name}_{i_pol}")

  # Coomputing NDWI = (NIR-SWIR)/(NIR+SWIR)
  # SWIR: Band 11
  # NIR: Band 8

  ndwi_num = (r_im[7] - r_im[10]) 
  ndwi_den = (r_im[7] + r_im[10]) 
  ndwi = np.divide(ndwi_num, ndwi_den, out= np.zeros_like(ndwi_num), where=ndwi_den!=0)
  plot_1_band(ndwi, f"{dir_path}/ndwi/{prefix}/{dir_name}_{i_pol}")

  # Coomputing MSAVI = (2*NIR-1-sqrt((2*NIR+1)^2-8*(NIR - RED)))/2
  # NIR: Band 8
  # RED: Band 4
  msavi = (2*r_im[7] + 1 - np.sqrt(np.power(2*r_im[7]+1,2) - 8 *(r_im[7] - r_im[3])))/2.
  plot_1_band(msavi, f"{dir_path}/msavi/{prefix}/{dir_name}_{i_pol}")

  # Coomputing Chlorophyll Index - Green  CI_GREEN = (NIR-GREEN)/(GREEN)
  # NIR: Band 8
  # GREEN: Band 3
  ci_green_num = (r_im[7]-r_im[2])
  ci_green_den = r_im[2]
  ci_green = np.divide(ci_green_num, ci_green_den, out= np.zeros_like(ci_green_num), where=ci_green_den!=0)
  plot_1_band(ci_green, f"{dir_path}/ci_green/{prefix}/{dir_name}_{i_pol}")

  # Coomputing Chlorophyll Index - Red Edge  CI_RED_EDGE = (NIR-RE)/(RE)
  # NIR: Band 8
  # RED-EDGE: Band 5
  ci_red_edge_num = (r_im[7]-r_im[4])
  ci_red_edge_den = r_im[4]
  ci_red_edge = np.divide(ci_red_edge_num, ci_red_edge_den, out= np.zeros_like(ci_red_edge_num), where=ci_red_edge_den!=0)
  plot_1_band(ci_red_edge, f"{dir_path}/ci_red_edge/{prefix}/{dir_name}_{i_pol}")

def main(argv=None):
  if argv == None:
    argv = sys.argv[1:]
  args = parse_command_line(argv)

  config_file = 'configs/default_parameters_sentinel.yaml'
  #config_file = 'configs/parameters_sentinel_1.yaml'
  with open(config_file, 'r') as file:
    inputs = yaml.safe_load(file)
  params = inputs['params']

  data_folder= params['request_params']['output_directory']
  print(data_folder)

  with open(f"{data_folder}/list_retrieved_images.txt", "r") as f:
    for line in f:
      dir_name = line.rstrip('\n')
      print(dir_name)
      reg, prov, com, code_com, period = dir_name.split('.')
      print(reg)
      print(prov)
      print(com)
      print(code_com)
      print(period)
      dir_path = f"{data_folder}/{dir_name}"
      json_file_name = f"data/GEOJSON/FEUDI/GEOJSON_FEUDI/{reg}.{prov}.{code_com}.{com}.geojson"
      gdf = gpd.read_file(json_file_name)
      polygons = gdf.geometry
      
      if os.path.isdir(f"{dir_path}"):
        print(line.rstrip('\n'))
      else:
        print(f"Image will be taken from {dir_path}.zip")
        os.system(f'mkdir {dir_path}')
        os.system(f'tar -xvzf {dir_path}.zip -C {dir_path}/')
      raster = rasterio.open(dir_path+"/default.tif")
      #print(raster.indexes)
      #print(raster.index(raster.bounds.left, raster.bounds.bottom))
      #print(raster.read(1))
      pathlib.Path(dir_path+"/ndvi/in").mkdir(parents=True, exist_ok=True)
      pathlib.Path(dir_path+"/ndwi/in").mkdir(parents=True, exist_ok=True)
      pathlib.Path(dir_path+"/msavi/in").mkdir(parents=True, exist_ok=True)
      pathlib.Path(dir_path+"/ci_green/in").mkdir(parents=True, exist_ok=True)
      pathlib.Path(dir_path+"/ci_red_edge/in").mkdir(parents=True, exist_ok=True)

      pathlib.Path(dir_path+"/ndvi/out").mkdir(parents=True, exist_ok=True)
      pathlib.Path(dir_path+"/ndwi/out").mkdir(parents=True, exist_ok=True)
      pathlib.Path(dir_path+"/msavi/out").mkdir(parents=True, exist_ok=True)
      pathlib.Path(dir_path+"/ci_green/out").mkdir(parents=True, exist_ok=True)
      pathlib.Path(dir_path+"/ci_red_edge/out").mkdir(parents=True, exist_ok=True)
      for i, poly in enumerate(polygons):
        masked_image, masked_transform = mask(raster, [poly.__geo_interface__], crop=True)
        masked_image_inverted, masked_transform = mask(raster, [poly.__geo_interface__], crop=True, invert=True)
        compute_parameters(masked_image,dir_path, dir_name, i, prefix="in")
        compute_parameters(masked_image_inverted,dir_path, dir_name, i, prefix="out")






if __name__== '__main__':
  status = main()
  sys.exit(status)
