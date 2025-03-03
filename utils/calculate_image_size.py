from pyproj import CRS, Transformer
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
    min_x, min_y = transformer.transform(float(bbox[0]), float(bbox[1]))
    max_x, max_y = transformer.transform(float(bbox[2]), float(bbox[3]))
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
