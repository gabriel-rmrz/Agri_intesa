import numpy as np
import pandas as pd
from shapely.geometry import mapping
from utils.transform_coord import transform_coord


def get_pixel_coord(pol, crs, res=0.2):
  pol_mapped = mapping(pol)['features'][0]['geometry']
  polygon_pixel_coord = []
  polygons = []
  if pol_mapped['type'] == 'Polygon':
    p = np.array(pol_mapped['coordinates'][0])
    poly_bounds = np.array([transform_coord(point, crs) for point in [ [np.min(p[:,0]), np.min(p[:,1])], [np.max(p[:,1]), np.max(p[:,1])]]])

    points = [ transform_coord(point, crs)  for point in pol_mapped['coordinates'][0]]
    point_ref = poly_bounds[0]
    polygon_pixel_coord.append([((int(point[0]/res)-int(point_ref[0]/res)), (int(poly_bounds[1][1]/res) - int(point_ref[1]/res)) - (int(point[1]/res) - int(point_ref[1]/res))) for point in points])
    polygons.append(p)
  if pol_mapped['type'] == 'MultiPolygon':
    for p in pol_mapped['coordinates'][0]:
      p = np.array(p)
      poly_bounds = np.array([transform_coord(point, crs) for point in [ [np.min(p[:,0]), np.min(p[:,1])], [np.max(p[:,1]), np.max(p[:,1])]]])
      points = [transform_coord(point, crs)  for point in p]
      point_ref = poly_bounds[0]
      polygon_pixel_coord.append([((int(point[0]/res)-int(point_ref[0]/res)), (int(poly_bounds[1][1]/res) - int(point_ref[1]/res)) - (int(point[1]/res) - int(point_ref[1]/res))) for point in points])
      polygons.append(p)
      #points_box = [transform_coord(point, crs)
  return polygons, polygon_pixel_coord

