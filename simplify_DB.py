import os
import re
import subprocess
from osgeo import ogr

def split_outside_parentheses(sentence):
  # This pattern matches words or parenthesis groups as a whole
  pattern = r'\([^)]*\)|\S+'
  matches = re.findall(pattern, sentence)
  return matches

def unpack_zip_files(PATH):
  # TODO: This part has to be adapte to unpack recursively.  DONE!
  paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] == '.zip']
  
  keep_unpacking = False 
  for p in paths:
    p_arr = p.rsplit("/")
    #print(p_arr[-1])
    dir_path = ".".join(p.rsplit(".")[:-1])
    #print(dir_path[0])
    if os.path.isdir(dir_path):
      print("")
      #print("the zip file has already been decompressed")
    else: 
      keep_unpacking = True
      bash_command = f"unzip ({dir_path}.zip) -d ({dir_path})"
      print(bash_command)
      print(bash_command.split())
      print(split_outside_parentheses(bash_command))
      matches = split_outside_parentheses(bash_command)
      cleaned = [m[1:-1] if m.startswith('(') and m.endswith(')') else m for m in matches]
  
      process = subprocess.Popen(cleaned, stdout=subprocess.PIPE)
      output, error = process.communicate()
  if keep_unpacking:
    unpack_zip_files(PATH)
  else:
    print("All zip files have been unpacked. No removing original zip files")
    for p in paths:
      os.remove(p)

       
    

def main():
  PATH='data/ITALIA'
  
  #unpack_zip_files(PATH)
  
  target_dir = "data/simplifiedDB"
  if os.path.isdir(target_dir):
    print("The simplified DB directory already exist")
    exit()
  else:
    os.mkdir(target_dir)
    path_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if f.split('_')[-1] == 'ple.gml']
    total_size = 0
    for pf in path_files:
      print(pf)
      total_size += os.path.getsize(pf)
      new_name = "_".join(pf.split("/")[2:]).split('_')
      del new_name[-3]
      del new_name[-1]
      new_name = target_dir + "/" + "_".join(new_name)+'.geojson'
      '''
      print(new_name)
      bash_command = f'ogr2ogr -f "GeoJSON" ({new_name}) ({pf})'
      print(bash_command)
      matches = split_outside_parentheses(bash_command)
      cleaned = [m[1:-1] if m.startswith('(') and m.endswith(')') else m for m in matches]
  
      process = subprocess.Popen(cleaned, stdout=subprocess.PIPE)
      output, error = process.communicate()
      # ogr2ogr -f "GeoJSON" test.geojson data/ITALIA/CAMPANIA/AV/A489_ATRIPALDA/A489_ATRIPALDA_ple.gml

  
      '''
  
      # Open source GM
      src_ds = ogr.Open(pf)
      src_layer = src_ds.GetLayer()
  
      # Create output GeoJSON
      driver = ogr.GetDriverByName("GeoJSON")
      out_ds = driver.CreateDataSource(new_name)
      out_layer = out_ds.CopyLayer(src_layer, src_layer.GetName())
  
      # Clean up
      del src_ds
      del out_ds
    print(total_size*(9.3132257461548e-10))

if __name__ == "__main__":
  main()
