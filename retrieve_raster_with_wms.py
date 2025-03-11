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

def main(argv=None):
  if argv == None:
    argv = sys.argv[1:]
  args = parse_command_line(argv, is_local=False)
  config_file = args['config_file']
  with open(config_file, 'r') as file:
    inputs = yaml.safe_load(file)
  params = inputs['params']
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
    print(in_file)
    # Extract filename without path
    filename = str(in_file).rsplit("/", 1)[-1]  # Get last segment after the last "/"

    # Remove extension
    result = filename.rsplit(".", 1)[0]
    print(result.split('.'))
    region, prov, cod_comune, comune = result.split('.')
    print(region)
    input_data = gpd.read_file(in_file, layer=comune)
    comune_pd = get_df(region,prov, cod_comune, comune)
    for fog, par, polygon_label in zip(input_data.Foglio, input_data.Particella, input_data.geometry):
      if fog is None:
        continue
      fog = re.sub(r'[^0-9]', '', fog)
      #print(f"{comune}      {fog}         {par}")
      print(f"comune: {comune}")
      #com_norm = normalize_string(comune) # Decompose accents
      cadastral_id = generate_cadastral_id(cod_comune, int(fog), int(par))
      print( cadastral_id)
      if not comune_pd.empty:
        polygon_all = comune_pd[comune_pd.gml_id == cadastral_id].geometry
        polygon_all = polygon_all.scale(1.03)
        precision = 5
        polygon_all = polygon_all.apply(lambda geom: round_geom_coords(geom, precision))
        #polygon_label = polygon_label.apply(lambda geom: round_geom_coords(geom, precision))
        #print(len(polygon_all['coordinates'][0]))
        if len(mapping(polygon_all)['features']) == 0:
          continue
        polygon_all_mapped = mapping(polygon_all)['features'][0]['geometry']#['coordinates'][0]

        
        '''
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        print(polygon_all_mapped['type'])
        if polygon_all_mapped['type'] == 'MultiPolygon':
          print(polygon_all)
          print("polygon_all_mapped['coordinates'][0]")
          print(polygon_all_mapped['coordinates'][0])
          print(polygon_all_mapped)
          exit()
        continue
        print(polygon_label)
        print(type(polygon_label))
        '''
        points = [Point((np.round(coord[0],5), np.round(coord[1],5))) for poly in polygon_label.geoms for coord in poly.exterior.coords]


        matches = []
        
        # Loop over polygon_all to assign index
        #print(polygon_all
        #for jdx, (idx, poly) in enumerate(polygon_all.items()):
        polygon_label = make_valid(polygon_label)
        if polygon_all_mapped['type'] == 'MultiPolygon':
          for jdx, poly in enumerate(polygon_all_mapped['coordinates'][0]):
            poly_intersection = polygon_label.intersection(Polygon(poly))
            if not poly_intersection.is_empty:
              matches.append(jdx) 
        else:
          poly = polygon_all_mapped['coordinates'][0]
          print("polygon_all")
          print(polygon_all)
          print("Polygon(points)")
          print(Polygon(points))
          print("polygon_all_mapped['type']")
          print(polygon_all_mapped['type'])
          poly_intersection = make_valid(polygon_all.item()).intersection(make_valid(Polygon(points)))
          print(poly_intersection)
          if not poly_intersection.is_empty:
            matches.append(0) 
        '''
        elif polygon_all_mapped['type'] == 'Polygon':
          poly = polygon_all_mapped['coordinates'][0]
          print("polygon_all")
          print(polygon_all)
          print("Polygon(points)")
          print(Polygon(points))
          print("polygon_all_mapped['type']")
          print(polygon_all_mapped['type'])
          poly_intersection = make_valid(polygon_all.item()).intersection(make_valid(Polygon(points)))
          print(poly_intersection)
          if not poly_intersection.is_empty:
            matches.append(0) 
        '''
        print(f'matches: {matches}')

          
        continue


        '''
        print(f"com_norm: {com_norm}")
        print(f"transform_string(comune): {transform_string(comune)}")
        print(f"cod_comuni: {info_comuni_df.head(100)}")
        print(f"cod_comuni: {info_comuni_df.comune.head(100)}")
        print(f"duplicated: {info_comuni_df[duplicated]}")
        filtered_comune_df = info_comuni_df[info_comuni_df['norm_comune'].str.contains(com_norm, regex=False)]
        print(f"filtered_comune_df: {filtered_comune_df}")
        print(f"filtered_comune_df.prefisso: {filtered_comune_df.prefisso}")
        
        if filtered_comune_df.cod_fisco.shape[0] == 1:
          print(f"str(filtered_comune_df.cod_fisco.item()): {str(filtered_comune_df.cod_fisco.item())}")
          print('Here')
        else:
          print(f"filtered_comune_df.cod_fisco.shape[0]: {filtered_comune_df.cod_fisco.shape[0]}")
          print(com_norm)
          print("Ups")
          print(f"gpd.list_layers(in_file): {gpd.list_layers(in_file)}")
          print(f"input_data.keys(): {input_data.keys()}")
          print(f"input_files: {input_files}")
          print(f"input_data.head(): {input_data.head()}")
          exit()
        
        break
        '''
        image_name = f"region_prov_{comune}_{fog}_{par}_{sub_poly}"
        get_images(params, image_name, polygon_label, plot_output=True)

  #make_bbox_geojson(input_files)


if __name__== "__main__":
  status = main()
  sys.exit(status)
