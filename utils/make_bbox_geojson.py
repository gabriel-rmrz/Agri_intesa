import geopandas as gpd

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
