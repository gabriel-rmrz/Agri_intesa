#from pyproj import CRS
import numpy as np
from utils.transform_coord import transform_coord
def get_pixel_coord_from_multipol(pol, crs, res=0.5):
  points_box = [transform_coord(point, crs) for point in [ (pol.bounds[0], pol.bounds[1]), (pol.bounds[2], pol.bounds[3])]]

  points = [ transform_coord(point, crs) for polygon in pol.geoms for point in polygon.exterior.coords[:-1]]
  point_ref = points_box[0]
  pixel_coord = [ ((int(point[0]/res)-int(point_ref[0]/res)), (int(points_box[1][1]/res) - int(point_ref[1]/res)) - (int(point[1]/res) - int(point_ref[1]/res))) for point in points]
  print(points_box)
  return np.array(points), pixel_coord
