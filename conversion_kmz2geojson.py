#!/usr/bin/env python

import os
import sys
import argparse
import zipfile
import geopandas as gdp
from pykml import parser
from shapely.geometry import Point, LineString, Polygon

def kmz_to_geojson(kmz_file_path, geojson_file_path):
  # Extracting the KMZ file
  with zipfile.ZipFile(kmz_file_path, 'r') as kmz:
    kmz.extractall("temp_kml")

  # Locate KML file
  kml_file_path = None
  for root, dirs, files in os.walk("temp_kml"):
    for file in files: 
      if file.endswith(".kml"):
        kml_file_path = os.path.join(root,file)
        break
  
  if not kml_file_path:
    raise FileNotFoundError("No KLM file found in the KMZ archive.")

  # Parse the KML data into GeoJSON
  data = gdp.read_file(kml_file_path)

  # Save as GeoJSON
  data.to_file(geojson_file_path, driver="GeoJSON")

  # Cleanup temporary files
  os.system("rm -rf temp_kml")
  print(f"GeoJSON file saved at {geojson_file_path}")

def parse_command_line(argv):
  parser = argparse.ArgumentParser(description="Conversion from compressed format kmz to geoJSON")
  parser.add_argument('-i','--input_name', help='Name of the input file in kmz format', required=True)
  parser.add_argument('-o','--output_name', help='Name of the output file in geoJSON format', nargs='?', default="output.geojson", type=str)
  return vars(parser.parse_args(argv))

def main(argv=None):
  if argv == None:
    argv = sys.argv[1:]
  args = parse_command_line(argv)
  print(args)
  input_file_name = args['input_name']
  output_file_name = args['output_name']
  #kmz_to_geojson("feudi_21256.kmz", "output.geojson")
  kmz_to_geojson(input_file_name, output_file_name)


if __name__== "__main__":
  status = main()
  sys.exit(status)
