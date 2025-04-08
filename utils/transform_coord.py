from pyproj import CRS, Transformer

def transform_coord(point_in, crs, res=0.5):
  # Create CRS object
  crs_obj = CRS.from_string(crs)

  # Check if the CRS uses degrees (geogrephic) or meters (projected)
  if crs_obj.is_geographic:
    # Transfrom bounding box to a projected CRS for meters
    transformer = Transformer.from_crs(crs_obj, CRS.from_epsg(3857), always_xy=True)
    point_out = transformer.transform(float(point_in[0]), float(point_in[1]))
  else:
    # Bounding box is already in projected coordinates (meters)
    point_out = point_in

  return point_out

