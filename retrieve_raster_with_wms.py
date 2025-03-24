#import os
import pathlib
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import re
import json
import unicodedata

from utils.parse_command_line import parse_command_line
from utils.transform_coord import transform_coord
from utils.calculate_image_size import calculate_image_size
from utils.make_bbox_geojson import make_bbox_geojson
from utils.get_pixel_coord_from_multipol import get_pixel_coord_from_multipol
from utils.generate_cadastral_id import generate_cadastral_id
from utils.get_df import get_df

from shapely.geometry import shape, Polygon, mapping, Point, MultiPolygon
from shapely.validation import make_valid
from PIL import Image as PILImage, ImageDraw
from IPython.display import Image
from io import BytesIO
from pyproj import CRS, Transformer
from owslib.wms import WebMapService

def normalize_string(s):
  if pd.isna(s):
    return ''
  nfkd_form = unicodedata.normalize('NFKD', s) # Decompose accents
  s_out = ''.join([c for c in nfkd_form if not unicodedata.combining(c)]).replace(' ', '_').replace('\'', '_').replace('__','_')
  s_out = re.sub(r'[^A-Za-z0-9_]', '', s_out).upper()
  return s_out

def transform_string(s):
  s_out = re.sub(r'[^A-Za-z0-9_ ]', '', s).upper()
  s_out = re.sub(r'[ ]', '_', s_out)
  return s_out

def get_images(params, image_name, polygon,plot_output=False):
  wms_url = params['wms_info']['url']
  wms = WebMapService(wms_url)
  crs_source= params['wms_info']['crs']
  crs_polygons= params['wms_info']['crs']
  res = params['wms_info']['resolution']

  p, p_pix = get_pixel_coord_from_multipol(polygon, crs_source, res)
  im_size = calculate_image_size(polygon.bounds, res, crs_source)

  img1 = wms.getmap(
    layers=params['wms_info']['layer_names'],
    #size= [600,400],
    size= im_size,
    srs= crs_source,
    bbox = polygon.bounds,
    #bbox = [14.7434, 40.8638, 15.1123, 41.0615],
    format= params['wms_info']['image_format'])

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

    ax1.set_title("Original Image")
    ax1.imshow(image_array)
    ax1.set_axis_off()

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("Masked Image")
    ax2.imshow(masked_image)
    ax2.set_axis_off()
    #fig.show()
    fig.savefig(f'{out_name}_masked.png')

def round_geom_coords(geom, precision):
    if geom.geom_type == 'Polygon':
        return Polygon(np.round(geom.exterior.coords, precision),
                       [np.round(interior.coords, precision) for interior in geom.interiors])
    elif geom.geom_type == 'MultiPolygon':
        return MultiPolygon([round_geom_coords(part, precision) for part in geom.geoms])
    elif geom.geom_type == 'Point':
        return Point(np.round(geom.coords[0], precision))
    else:
        raise ValueError(f'Geometry type {geom.geom_type} not supported')

def get_label_pixel_coord(polygon_data, polygon_label, params):
  pass
def get_label_mask(params, mask_file_name, polygon_data, polygon_intersection):
  res = params['wms_info']['resolution']
  p_data = np.array(polygon_data)
  p_intersection = np.array(polygon_intersection)
  image_bounds = np.array((float(np.min(p_data[:,0])), float(np.min(p_data[:,1])), float(np.max(p_data[:,0])), float(np.max(p_data[:,1]))))
  p_intersection_pixel = np.copy(p_intersection)
  p_intersection_pixel[:,0] = ((p_intersection_pixel[:,0] - image_bounds[0])/res).astype(int)
  p_intersection_pixel[:,1] = ((p_intersection_pixel[:,1] - image_bounds[1])/res).astype(int)
  p_intersection_pixel = p_intersection_pixel.astype(int)
  p_intersection_pixel_dict = {'pixel_polygon': p_intersection_pixel.tolist()}
  print(p_intersection_pixel_dict)
  
  with open(f'samples/mask_{mask_file_name}.json', 'w') as fp:
    json.dump(p_intersection_pixel_dict, fp)

  image_bounds_pixel = np.copy(image_bounds)
  image_bounds_pixel[2] = ((image_bounds_pixel[2] - image_bounds_pixel[0])/res).astype(int)
  image_bounds_pixel[3] = ((image_bounds_pixel[3] - image_bounds_pixel[1])/res).astype(int)
  image_bounds_pixel[0] = 0
  image_bounds_pixel[1] = 0
  image_bounds_pixel = image_bounds_pixel.astype(int)
  mask = np.zeros((image_bounds_pixel[2], image_bounds_pixel[3]), dtype=np.uint8)
  mask_image = PILImage.fromarray(mask)
  draw = ImageDraw.Draw(mask_image)
  draw.polygon(p_intersection_pixel, fill=1)
  mask = np.array(mask_image)
  print(f"samples/array_mask_{mask_file_name}")
  np.savez_compressed(f"samples/array_mask_{mask_file_name}", mask)

def main(argv=None):
  if argv == None:
    argv = sys.argv[1:]
  args = parse_command_line(argv, is_local=False)
  config_file = args['config_file']
  with open(config_file, 'r') as file:
    inputs = yaml.safe_load(file)
  params = inputs['params']
  crs_labels =  params['wms_info']['crs']
  crs_data =  params['requests_file']['crs']
  crs_meters = CRS.from_epsg(3857)
  transformer_label = Transformer.from_crs(CRS(crs_labels), crs_meters, always_xy=True)
  transformer_data = Transformer.from_crs(CRS(crs_data), crs_meters, always_xy=True)
  parcels_file = params['requests_file']['path']
  requests_pd = pd.read_excel(parcels_file,
                             index_col=0,
                             dtype = params['requests_file']['format'])
  
  #wms = WebMapService(params['wms']['url'])
  input_files = [f for f in pathlib.Path().glob("../GEOJSON/FEUDI/GEOJSON_FEUDI/*.geojson")]

  with open('data/italy_cities.json', 'r') as file:
    info_comuni_df = pd.DataFrame(json.load(file)['Foglio1'])
  duplicated = info_comuni_df['comune'].duplicated()
  info_comuni_df['norm_comune'] = info_comuni_df['comune'].apply(normalize_string)

  for in_file in input_files:
    # Extract filename without path
    filename = str(in_file).rsplit("/", 1)[-1]  # Get last segment after the last "/"

    # Remove extension
    result = filename.rsplit(".", 1)[0]
    region, prov, cod_comune, comune = result.split('.')
    input_data = gpd.read_file(in_file, layer=comune)
    comune_pd = get_df(region,prov, cod_comune, comune)


    for fog, par, polygon_label in zip(input_data.Foglio, input_data.Particella, input_data.geometry):
      print(f"foglio: {fog};     particella:{par}")
      if fog is None:
        continue
      fog = re.sub(r'[^0-9]', '', fog)
      #com_norm = normalize_string(comune) # Decompose accents
      cadastral_id = generate_cadastral_id(cod_comune, int(fog), int(par))
      if not comune_pd.empty:
        print("Comune non empty")
        polygon_all = comune_pd[comune_pd.gml_id == cadastral_id].geometry
        polygon_all = polygon_all.scale(1.03)
        precision = 6
        polygon_all = polygon_all.apply(lambda geom: round_geom_coords(geom, precision))
        #polygon_label = polygon_label.apply(lambda geom: round_geom_coords(geom, precision))
        if len(mapping(polygon_all)['features']) == 0:
          continue
        print("Geometry present")
        polygon_all_mapped = mapping(polygon_all)['features'][0]['geometry']#['coordinates'][0]
        #points = [Point((np.round(coord[0],5), np.round(coord[1],5))) for poly in polygon_label.geoms for coord in poly.exterior.coords]
        matches = []
        # Loop over polygon_all to assign index
        polygon_label = make_valid(polygon_label)
        polygon_label_mapped = mapping(polygon_label)#['features'][0]['geometry']

        if polygon_label_mapped['type'] == 'GeometryCollection':
          polygon_label_transformed = [transformer_label.transform(c[0],c[1]) for c in polygon_label_mapped['geometries'][0]['coordinates'][0]]
        else:
          depth = lambda L: isinstance(L, tuple) and max(map(depth, L))+1
          if depth(polygon_label_mapped['coordinates'][0]) == 2:
            polygon_label_mapped = polygon_label_mapped['coordinates'][0]
          elif depth(polygon_label_mapped['coordinates'][0]) == 3:
            polygon_label_mapped = polygon_label_mapped['coordinates'][0][0]

          polygon_label_transformed = [transformer_label.transform(c[0],c[1]) for c in polygon_label_mapped]


        if polygon_all_mapped['type'] == 'MultiPolygon':
          for jdx, poly in enumerate(polygon_all_mapped['coordinates'][0]):
            #TODO: Very important to change the coordinates before the intersection!!!!!!!!!!!
            poly_transformed = [transformer_data.transform(c[0],c[1]) for c in poly]
            poly_intersection = Polygon(polygon_label_transformed).intersection(Polygon(poly_transformed))
            if not poly_intersection.is_empty:
              poly_intersection = mapping(poly_intersection)['coordinates'][0]
              matches.append(jdx) 
              image_name = f"label_region_prov_{comune}_{fog}_{par}_{jdx}"
              get_label_mask(params, image_name, poly_transformed, poly_intersection)
              print(image_name)
        else:
          poly = polygon_all_mapped['coordinates'][0]
          poly_transformed = [transformer_data.transform(c[0],c[1]) for c in poly]

          #poly_intersection = make_valid(polygon_all.item()).intersection(make_valid(Polygon(points)))
          poly_intersection = make_valid(Polygon(poly_transformed)).intersection(make_valid(Polygon(polygon_label_transformed)))
          if not poly_intersection.is_empty:
            poly_intersection = mapping(poly_intersection)['coordinates'][0]
            matches.append(0) 
            image_name = f"label_region_prov_{comune}_{fog}_{par}_0"
            get_label_mask(params, image_name, poly_transformed, poly_intersection)
          

    #get_images(params, image_name, polygon_label, plot_output=True)

  #make_bbox_geojson(input_files)


if __name__== "__main__":
  status = main()
  sys.exit(status)
