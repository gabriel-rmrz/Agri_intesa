#import os
import pathlib
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from PIL import Image as PILImage, ImageDraw
from IPython.display import Image
from io import BytesIO
from pyproj import CRS, Transformer
from owslib.wms import WebMapService

def parse_command_line(argv):
  parser = argparse.ArgumentParser(description="Conversion from compressed format kmz to geoJSON")
  parser.add_argument('-i','--config_file', help='Name of the input file in kmz format', nargs='?', default='configs/default_parameters.yaml', type=str, required=True)
  #parser.add_argument('-o','--output_name', help='Name of the output file in geoJSON format', nargs='?', default="output.geojson", type=str)
  return vars(parser.parse_args(argv))

def transform_coord(point_in, crs, res=0.5):
  """
  Calculate image size for a WMS request based on bounding box and max resolution
  """
  # Create CRS object
  crs_obj = CRS.from_string(crs)

  # Check if the CRS uses degrees (geogrephic) or meters (projected)
  if crs_obj.is_geographic:
    # Transfrom bounding box to a projected CRS for meters
    transformer = Transformer.from_crs(crs_obj, CRS.from_epsg(3857), always_xy=True)
    point_out = transformer.transform(point_in[0], point_in[1])
  else:
    # Bounding box is already in projected coordinates (meters)
    point_out = point_in

  return point_out

def get_pixel_coord_from_multipol(pol, crs, res=0.5):
  points_box = [transform_coord(point, crs) for point in [ (pol.bounds[0], pol.bounds[1]), (pol.bounds[2], pol.bounds[3])]]

  points = [ transform_coord(point, crs) for polygon in pol.geoms for point in polygon.exterior.coords[:-1]]
  point_ref = points_box[0]
  pixel_coord = [ ((int(point[0]/res)-int(point_ref[0]/res)), (int(points_box[1][1]/res) - int(point_ref[1]/res)) - (int(point[1]/res) - int(point_ref[1]/res))) for point in points]
  print(points_box)
  return pixel_coord

def calculate_image_size(bbox, max_resolution, crs):
  """
  Calculate image size for a WMS request based on bounding box and max resolution
  """
  # Create CRS object
  crs_obj = CRS.from_string(crs)

  # Check if the CRS uses degrees (geogrephic) or meters (projected)
  if crs_obj.is_geographic:
    # Transfrom bounding box to a projected CRS for meters
    transformer = Transformer.from_crs(crs_obj, CRS.from_epsg(3857), always_xy=True)
    min_x, min_y = transformer.transform(bbox[0], bbox[1])
    max_x, max_y = transformer.transform(bbox[2], bbox[3])
  else:
    # Bounding box is already in projected coordinates (meters)
    min_x, min_y, max_x, max_y = bbox

  #Calculate spatial extents in meters
  extent_x = max_x - min_x
  extent_y = max_y - min_y

  # Calculate image dimensions
  width = int(extent_x / max_resolution)
  height = int(extent_y / max_resolution)

  return width, height

def get_images(wms, layer_name, input_data,plot_output=False):
    print(input_data.head)
    for id_parcel, pol in zip(input_data.id, input_data.geometry):
        crs='EPSG:4326'
        max_res = 0.2

        polygon_coords = get_pixel_coord_from_multipol(pol, crs, max_res)

        print(list(pol.geoms))
        print(list(pol.bounds))

        im_size = calculate_image_size(pol.bounds, max_res, crs)
        print(im_size)
        img1 = wms.getmap(
            layers=['0'],
            #size= [600,400],
            size= im_size,
            srs= "EPSG:4326",
            bbox = list(pol.bounds),
            #bbox = [14.7434, 40.8638, 15.1123, 41.0615],
            format= "image/jpeg")
        print(type(Image(img1)))

        Image(img1.read())
        out_name = f'samples/sample_{layer_name}_id_{id_parcel}'
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
        draw.polygon(polygon_coords, fill = 1)
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
  args = parse_command_line(argv)
  config_file = args['config_file']
  with open(config_file, 'r') as file:
    inputs = yaml.safe_load(file)
  
  print(inputs)
  wms = WebMapService(inputs['params']['wms']['url'])
  input_files = [f for f in pathlib.Path().glob("../GEOJSON/FEUDI/GEOJSON_FEUDI/*.geojson")]
  for in_file in input_files:
    layer_name = gpd.list_layers(in_file).name[0]
    input_data = gpd.read_file(in_file, layer=layer_name)
    get_images(wms, layer_name, input_data, True)



if __name__== "__main__":
  status = main()
  sys.exit(status)
