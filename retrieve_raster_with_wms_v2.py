TEST = True
#import os
import time
import urllib
import json
import pathlib
import sys
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import cv2
from pathlib import Path
from shapely.geometry import shape
from PIL import Image as PILImage, ImageDraw
from IPython.display import Image
from io import BytesIO
from pyproj import CRS, Transformer
from owslib.wms import WebMapService
from datetime import datetime

def parse_command_line(argv):
  parser = argparse.ArgumentParser(description="Conversion from compressed format kmz to geoJSON")
  parser.add_argument('-i','--config_file', help='Name of the input file in kmz format', nargs='?', default='configs/default_parameters.yaml', type=str, required=True)
  #parser.add_argument('-o','--output_name', help='Name of the output file in geoJSON format', nargs='?', default="output.geojson", type=str)
  return vars(parser.parse_args(argv))


def calculate_image_size(bounds, max_resolution, crs):
  """
  Calculate image size for a WMS request based on bounding box and max resolution
  """
  # Create CRS object
  crs_obj = CRS.from_string(crs)

  # Check if the CRS uses degrees (geogrephic) or meters (projected)
  if crs_obj.is_geographic:
    # Transfrom bounding box to a projected CRS for meters
    transformer = Transformer.from_crs(crs_obj, CRS.from_epsg(3857), always_xy=True)
    min_x, min_y = transformer.transform(bounds[0], bounds[1])
    max_x, max_y = transformer.transform(bounds[2], bounds[3])
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

def get_image(source_url_wms, source_crs, source_res, province_name, foglio, particella, catasto_bbox, catasto_polygon_pixel, catasto_crs='EPSG:6706', catasto_res=0.02, plot_output=True):
  catasto_bounds = [catasto_bbox[0][0], catasto_bbox[0][1], catasto_bbox[2][0], catasto_bbox[2][1] ]
  source_polygon_pixel = (catasto_res*catasto_polygon_pixel/source_res).astype(int)
  print("----catasto_bounds----")
  print(catasto_bounds)
  print("----catasto_bbox----")
  print(catasto_bbox)
  print("----source_polygon_pixel----")
  print(source_polygon_pixel)

  transformer = Transformer.from_crs(catasto_crs, source_crs, always_xy=True)

  source_bounds_min_x, source_bounds_min_y = transformer.transform(catasto_bounds[0], catasto_bounds[1]) 
  source_bounds_max_x, source_bounds_max_y = transformer.transform(catasto_bounds[2], catasto_bounds[3]) 
  #TODO: source bounding box in source_crs
  source_bounds = [source_bounds_min_x, source_bounds_min_y, source_bounds_max_x, source_bounds_max_y]
  print("----source_bounds----")
  print(source_bounds)
  exit()

  im_size = [np.max(source_polygon_pixel[:,0]), np.max(source_polygon_pixel[:,1])]
  print("----im_size----")
  print(im_size)
  print("----source_polygon_pixel[:,0]----")
  print(source_polygon_pixel[:,0])
  wms = WebMapService(source_url_wms)
  img1 = wms.getmap(
    layers=['0'],
    size= im_size,
    srs= source_crs,
    bbox = source_bounds,
    #bbox = [14.7434, 40.8638, 15.1123, 41.0615],
    format= "image/jpeg")
  print(type(Image(img1)))

  Image(img1.read())
  out_name = f'samples/sample_{province_name}_foglio_{foglio}_particella_{particella}'
  out = open(f'{out_name}.png', 'wb')
  out.write(img1.read())

  # Convert IPython Image to Numpy array
  image_data = BytesIO(Image(img1.read()).data)
  pil_image = PILImage.open(image_data).convert('RGB')
  image_array = np.array(pil_image)

  # Create a mask
  mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
  print("-----image_array.shape-----")
  print(image_array.shape)
  mask_image = PILImage.fromarray(mask)
  draw = ImageDraw.Draw(mask_image)
  source_polygon_pixel = [(x,y) for x,y in source_polygon_pixel]
  draw.polygon(source_polygon_pixel, fill = 1)
  mask = np.array(mask_image)
  print("-----mask-----")
  print(mask)
  print("----source_polygon_pixel----")
  print(source_polygon_pixel)

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
  exit()

def make_bbox_geojson(input_files):
  province_bbox = {}
  province_bbox['type']= 'FeatureCollection'
  #province_bbox['name']= 'AllRegions'
  #province_bbox['crs']= {'type': 'name', 'properties': {'name': "urn:ogc:def:crs:OGC:1.3:CRS84"}}
  features = []
  for id_, in_file in enumerate(input_files):
    layer_name = gpd.list_layers(in_file).name[0]
    input_data = gpd.read_file(in_file, layer=layer_name)
    print(input_data.head)
    print(input_data.geometry.bounds.minx)
    print(input_data.geometry.bounds.minx.min())
    min_x_prov = input_data.geometry.bounds.minx.min().item()
    min_y_prov = input_data.geometry.bounds.miny.min().item()
    max_x_prov = input_data.geometry.bounds.maxx.max().item()
    max_y_prov = input_data.geometry.bounds.maxy.max().item()
    features.append(
      {
        "id":int(id_),
        "type": "Feature",
        "properties": {},
        "geometry":{
          "type": "Polygon",
          "coordinates": [
            [
              [min_x_prov, min_y_prov],
              [max_x_prov, min_y_prov],
              [max_x_prov, max_y_prov],
              [min_x_prov, max_y_prov],
              [min_x_prov, min_y_prov]
            ]
          ]
        }
      }
    )
  province_bbox['features']=features
  geometries = [shape(feature["geometry"]) for feature in features]
  properties = [feature["properties"] for feature in features]
  ids = [feature["id"] for feature in features]
  gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
  gdf["id"] = ids
  #print(province_bbox)
  #gdf = gpd.GeoDataFrame(province_bbox, crs="EPSG:4326")
  print(gdf.to_json())
  gdf.to_file('all_bbox_limits.geojson', driver='GeoJSON')

def get_token(max_tries=3):
  for i in range(1,max_tries+1):
    print(f"Token request. Try {i}/{max_tries}...")
    url_token = "https://catastomappe.it/get_token.php"
    token_request = urllib.request.urlopen(url_token)
    token = [t for t in token_request]
    if len(token) == 0:
      time.sleep(5)
      continue
    token = token[0].decode('utf-8')
    token_json = {
        "token": token,
        "issued_time": datetime.timestamp(datetime.now())
        }
    #token_json = list([token_json])
    token_file_name= f"configs/token.json"
    print(f"Saving token into {token_file_name}.")
    with open(token_file_name, "w") as token_file:
      json.dump(token_json, token_file)
    return token
  return None

def get_image_from_bbox(bbox, crs= 'EPSG:4326', res = 0.02,  url="https://wms.cartografia.agenziaentrate.gov.it/inspire/wms/ows01.php"):
    bounds = [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]]

    bb_wms = WebMapService(url) # ,version = "1.3.0") # To add attributes
    print(bb_wms.contents)
    #crs='EPSG:6706'
    max_res = 0.02
    #polygon_coords = get_pixel_coord_from_multipol(bounds, crs, max_res)
    im_size = calculate_image_size(bounds, max_res, crs)
    img1 = bb_wms.getmap(
        layers=['CP.CadastralParcel'],
        #size= [600,400],
        size= im_size,
        srs= crs,
        bbox = list(bounds),
        #bbox = [14.7434, 40.8638, 15.1123, 41.0615],
        format= "image/jpeg")
    print("...")
    print(type(Image(img1)))
    print("...")
    #Image(img1.read())
    out_name = f'samples/test_particella'
    out = open(f'{out_name}.png', 'wb')
    out.write(img1.read())
    print("image saved...")
    return img1


# Assume 'img_display' is an IPython.core.display.Image object
def detect_polygon(in_image, output_path="output.jpg", brightness_threshold=80):
    # Convert IPython Image to NumPy array
    #print(f'detect_polygons: {type(in_image.data)}')
    #img_bytes = BytesIO(in_image)
    #img_bytes = in_image
    image_data = BytesIO(Image(in_image.read()).data)
    pil_image = PILImage.open(image_data).convert('RGB')
    #pil_image = Image(img_bytes).convert("RGB")
    image = np.array(pil_image)
    bbox_im = np.array([0, 0,image.shape[0], image.shape[1]])
    print(f"image.shape: {image.shape}")
    print(f"bbox_im: {bbox_im}")


    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)


    # Apply Canny edge detection
    edges = cv2.Canny(gray, 200, 250)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Approximate contours to polygons and draw them
    min_diff =1000
    poly = None
    for contour in contours:
        epsilon = 0.002 * cv2.arcLength(contour, True)  # Precision of approximation
        approx = cv2.approxPolyDP(contour, epsilon, True)
        bbox_pol = np.array(cv2.boundingRect(approx))
        print(f'bbox_pol: {bbox_pol}')
        bbox_diff = np.sum(np.abs(bbox_im-bbox_pol))
        print(f'bbox_diff: {bbox_diff}')
        # Draw polygon if it has at least 3 sides
        if len(approx) >= 3 and bbox_diff < min_diff :
            poly = approx
            min_diff = bbox_diff
            #cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)  # Green polygons
    if np.all(poly) != None:
        cv2.drawContours(image, [poly], 0, (0, 255, 0), 2)  # Green polygons
    # Save the output image
    cv2.imwrite(output_path, image)

    # Display the result in Jupyter Notebook
    Image(filename=output_path)
    print(type(poly))
    print(poly)
    return poly

def make_request(url_request_incomple):
  token = None
  if Path("configs/token.json").is_file():
    print("Reading token file.")
    with open("configs/token.json", 'r') as file:
      token_from_json = json.load(file)
    print("Checking token validity")
    if (datetime.timestamp(datetime.now()) - token_from_json['issued_time']) < 300:
      token = token_from_json['token']
    else:
      print("The token in the file is not longer valid. Trying other options")
  if token == None:
    token = get_token()
  if token == None:
    print("Token cannot be issued at this moment. Please retry again later.")
    return None

  url_request = url_request_incomple + token
  print("Making request to the the Geoportale Cartografico Catastale")
  bbox_request = None
  try:
    bbox_request = urllib.request.urlopen(url_request)
  except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code} - {e.reason}")
  except urllib.error.URLError as e:
    print(f"URL Error: {e.reason}")
  except Exception as e:
    print(f"Unexpected Error: {e}")
  return bbox_request
  
def get_polygon_from_ae():
  #TODO: Create a json file or yml with the info of the parcels of interest.
  prov = "AV"
  url_agenzia = "https://wms.cartografia.agenziaentrate.gov.it"
  cod_comune = "L589"
  foglio=1
  num_part = "00003"
  url_request_incomple = f"{url_agenzia}/inspire/ajax/ajax.php?op=getGeomPart&prov={prov}&cod_com={cod_comune}&foglio={foglio}&num_part={num_part}&tkn="
  bbox_request = make_request(url_request_incomple)
  if bbox_request == None:
    print("Request cannot be compleated at the moment")
    exit()

  print(bbox_request)
  geom_all = [json.loads(bb) for bb in bbox_request]
  print(geom_all)
  if TEST:
    bbox = [ [15.29082657, 41.07571757], [15.29107614, 41.07571757], [15.29107614, 41.07588641], [15.29082657, 41.07588641], [15.29082657, 41.07571757] ]
  else:
    try:
      print(type(geom_all[0]))
      print(geom_all[0])
      if isinstance(geom_all[0], str):
        json_data = json.loads(geom_all[0])

      # Check if the parsed data is a dictionary
    except json.JSONDecodeError:
      print("Request cannot be compleated at the moment")
      exit()
    geom = json.loads(geom_all[0]['GEOMETRIA'][0])
    #print(type(geom["coordinates"][0][0]))
    print("Request successful!")
    bbox=geom["coordinates"][0]
    print(bbox)

  
  #get_image_from_bbox(bbox,"https://siat2.provincia.avellino.it/server/services/RASTER/Ortofoto_AGEA_2020/MapServer/WMSServer")
  img_catas = get_image_from_bbox(bbox,'EPSG:6706', 0.02)
  polygon = detect_polygon(img_catas, "black_polygons.jpg", brightness_threshold=200)
  polygon =  polygon.reshape((polygon.shape[0], polygon.shape[2]))
  return bbox, polygon

def main(argv=None):
  if argv == None:
    argv = sys.argv[1:]
  args = parse_command_line(argv)
  config_file = args['config_file']
  particelle_pd = pd.read_excel("data/feudi.xlsx", 
                                index_col=0, 
                                dtype={'id':int, 
                                       'Comune':str, 
                                       'Foglio': int,
                                       'Particella': int, 
                                       'Pagina':int,
                                       'Ha': int,
                                       'Aa':int,
                                       'Ca':int}
                                )
  print(particelle_pd)
  with open(config_file, 'r') as file:
    inputs = yaml.safe_load(file)
  
  catasto_bbox, catasto_polygon_pixel = get_polygon_from_ae()

  source_url_wms =inputs['params']['wms']['url']
  source_crs = "EPSG:4326"
  source_res = 0.2
  province_name = "province_test"
  catasto_crs='EPSG:6706'
  catasto_res = 0.02
  foglio = 'foglio_test'
  particella = 'paritcella_test'


  get_image(source_url_wms, source_crs, source_res, province_name,foglio, particella, catasto_bbox, catasto_polygon_pixel, catasto_crs, catasto_res, plot_output=True)
  exit()

  make_bbox_geojson(input_files)


if __name__== "__main__":
  status = main()
  sys.exit(status)
