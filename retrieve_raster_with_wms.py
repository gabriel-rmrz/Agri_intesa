#import os
import pathlib
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

from utils.parse_command_line import parse_command_line
from utils.transform_coord import transform_coord
from utils.calculate_image_size import calculate_image_size
from utils.make_bbox_geojson import make_bbox_geojson
from utils.get_pixel_coord_from_multipol import get_pixel_coord_from_multipol

from shapely.geometry import shape, Polygon, mapping
from PIL import Image as PILImage, ImageDraw
from IPython.display import Image
from io import BytesIO
from pyproj import CRS, Transformer
from owslib.wms import WebMapService

def get_images(params, image_name, polygon,plot_output=False):
  wms_url = params['wms_info']['url']
  wms = WebMapService(wms_url)
  crs_source= params['wms_info']['crs']
  crs_polygons= params['wms_info']['crs']
  res = params['wms_info']['resolution']

  p, p_pix = get_pixel_coord_from_multipol(polygon, crs_source, res)
  im_size = calculate_image_size(polygon.bounds, res, crs_source)
  print(im_size)




  img1 = wms.getmap(
    layers=params['wms_info']['layer_names'],
    #size= [600,400],
    size= im_size,
    srs= crs_source,
    bbox = polygon.bounds,
    #bbox = [14.7434, 40.8638, 15.1123, 41.0615],
    format= params['wms_info']['image_format'])
  print(type(Image(img1)))

  Image(img1.read())
  out_name = f'samples/sample_{image_name}'
  out = open(f'{out_name}.png', 'wb')
  out.write(img1.read())



  # Convert IPython Image to Numpy array
  image_data = BytesIO(Image(img1.read()).data)
  pil_image = PILImage.open(image_data).convert('RGB')
  image_array = np.array(pil_image)

  # Create a mask
  mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
  mask_image = PILImage.fromarray(mask)
  draw = ImageDraw.Draw(mask_image)
  draw.polygon(p_pix, fill = 1)
  mask = np.array(mask_image)

  # Apply the mask to the image)
  masked_image = np.copy(image_array)
  masked_image[mask!=1] = (0,0,0) # Setmasked pixels to black

  if plot_output:
    # Display the original and masked images

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    print(type(ax1))

    ax1.set_title("Original Image")
    ax1.imshow(image_array)
    ax1.set_axis_off()

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("Masked Image")
    ax2.imshow(masked_image)
    ax2.set_axis_off()

    #fig.show()

    fig.savefig(f'{out_name}_masked.png')


def main(argv=None):
  if argv == None:
    argv = sys.argv[1:]
  args = parse_command_line(argv, is_local=False)
  config_file = args['config_file']
  with open(config_file, 'r') as file:
    inputs = yaml.safe_load(file)
  params = inputs['params']
  
  #wms = WebMapService(params['wms']['url'])
  input_files = [f for f in pathlib.Path().glob("../GEOJSON/FEUDI/GEOJSON_FEUDI/*.geojson")]

  for in_file in input_files:
    comune = gpd.list_layers(in_file).name[0]
    input_data = gpd.read_file(in_file, layer=comune)
    print(comune)
    for fog, par, polygon in zip(input_data.Foglio, input_data.Particella, input_data.geometry):
      print(f"{comune}      {fog}         {par}")
      image_name = f"region_prov_{comune}_{fog}_{par}"
      get_images(params, image_name, polygon, plot_output=True)
    exit()

  make_bbox_geojson(input_files)


if __name__== "__main__":
  status = main()
  sys.exit(status)
