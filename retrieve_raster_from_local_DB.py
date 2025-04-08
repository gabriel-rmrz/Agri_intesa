
import sys
import yaml
import argparse
import re
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.parse_command_line import parse_command_line
from utils.transform_coord import transform_coord
from utils.calculate_image_size import calculate_image_size
from utils.get_df import get_df
from utils.generate_cadastral_id import generate_cadastral_id
from utils.get_pixel_coord import get_pixel_coord

from owslib.wms import WebMapService
from shapely.geometry import Polygon, mapping
from shapely.errors import GEOSException
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, mapping
from IPython.display import Image
from io import BytesIO
from PIL import Image as PILImage, ImageDraw

def transform_string(s):
  s_out = re.sub(r'[^A-Za-z0-9_ ]', '', s).upper()
  s_out = re.sub(r'[ ]', '_', s_out)
  return s_out 

def get_images(params, image_name, polygon, plot_output=False):
  wms_url =   params['wms_info']['url']
  wms = WebMapService(wms_url)
  crs_source = params['wms_info']['crs']
  crs_polygons = params['requests_file']['crs']
  res = params['wms_info']['resolution']

  polygons, polygon_coords = get_pixel_coord(polygon, crs_polygons, res)
  for j, (p, p_pix) in  enumerate(zip(polygons, polygon_coords)):
    
    transformer = Transformer.from_crs(crs_polygons, crs_source, always_xy=True)
    
    image_bounds_min = transformer.transform(float(np.min(p[:,0])),float(np.min(p[:,1])))
    image_bounds_max = transformer.transform(float(np.max(p[:,0])),float(np.max(p[:,1])))
    image_bounds = tuple((image_bounds_min[0], image_bounds_min[1], image_bounds_max[0], image_bounds_max[1]))
    print(type(image_bounds))
    print(image_bounds)




    im_size = calculate_image_size(image_bounds, res, crs_source)

    img1 = wms.getmap(
      layers=params['wms_info']['layer_names'],
      size= im_size,
      srs= crs_source,
      #bbox = list([ image_bounds_min[0], image_bounds_min[1], image_bounds_max[0], image_bounds_max[1]]),
      bbox = image_bounds,
      format= params['wms_info']['image_format'])


    # Convert IPython Image to Numpy array
    image_data = BytesIO(Image(img1.read()).data)
    pil_image = PILImage.open(image_data).convert('RGB')
    image_array = np.array(pil_image)
    np.savez_compressed(f"samples/array_{image_name}.npz",image_array)
    print(f"samples/array_{image_name}.npz")
    # Create a mask
    mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
    mask_image = PILImage.fromarray(mask)
    draw = ImageDraw.Draw(mask_image)
    draw.polygon(p_pix, fill = 1)
    mask = np.array(mask_image)
    np.savez_compressed(f"samples/array_mask_{image_name}.npz",mask)

    if plot_output:
      #Image(img1.read())
      out_name = f'samples/{image_name}_{j}'
      out = open(f'{out_name}.png', 'wb')
      out.write(img1.read())
      # Apply the mask to the image)
      masked_image = np.copy(image_array)
      masked_image[mask!=1] = (0,0,0) # Setmasked pixels to black
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
  for region in provinces.keys():
    for prov in provinces[region]:
      req_prov_pd = requests_pd.query(f'Regione=="{region}" and SiglaProvincia == "{prov}"')
      req_com_pd = req_prov_pd[~req_prov_pd["CodComune"].duplicated()][['CodComune', 'Comune']]
      for i, c in req_com_pd.iterrows():
        comune_pd = get_df(region,prov, c.CodComune, c.Comune)
        #print(f'{region}-{prov}-Retrieving images for the "comune": {c.Comune}')

        if comune_pd.empty:
          continue
        for foglio, particella in zip(req_prov_pd[req_prov_pd.CodComune == c.CodComune].Foglio, req_prov_pd[req_prov_pd.CodComune == c.CodComune].Particella):
          #print(f"Foglio {foglio}. Retrieving parcel {particella}")

          cadastral_id = generate_cadastral_id(c.CodComune, foglio, particella)
          image_name = f"{region}_{prov}_{transform_string(c.Comune)}_{foglio}_{particella}"
          print(f"transform_string(c.Comune): {transform_string(c.Comune)}")
          polygon = comune_pd[comune_pd.gml_id == cadastral_id].geometry
          if polygon.shape[0] == 1:
            get_images(params, image_name, polygon, True)

if __name__=="__main__":
  status = main()
  sys.exit(status)
