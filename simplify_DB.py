import os
import re
import subprocess

def split_outside_parentheses(sentence):
  # This pattern matches words or parenthesis groups as a whole
  pattern = r'\([^)]*\)|\S+'
  matches = re.findall(pattern, sentence)
  return matches

PATH='data/ITALIA'


# TODO: This part has to be adapte to unpack recursively.
paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] == '.zip']

for p in paths:
  p_arr = p.rsplit("/")
  #print(p_arr[-1])
  dir_path = "".join(p.rsplit(".")[:-1])
  #print(dir_path[0])
  if os.path.isdir(dir_path):
    print("")
    #print("the zip file has already been decompressed")
  else: 
    bash_command = f"unzip ({dir_path}.zip) -d ({dir_path})"
    print(bash_command)
    print(bash_command.split())
    print(split_outside_parentheses(bash_command))
    matches = split_outside_parentheses(bash_command)
    cleaned = [m[1:-1] if m.startswith('(') and m.endswith(')') else m for m in matches]

    process = subprocess.Popen(cleaned, stdout=subprocess.PIPE)
    output, error = process.communicate()

target_dir = "data/simplifiedDB"
if os.path.isdir(target_dir):
  print("The simplified DB directory already exist")
  exit()
else:
  os.mkdir(target_dir)
  path_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if f.split('_')[-1] == 'ple.gml']
  for pf in path_files:
    print(pf)
    new_name = "_".join(pf.split("/")[2:]).split('_')
    del new_name[-3]
    del new_name[-1]
    new_name = "_".join(new_name)+'.json'
    print(new_name)


    # Open source GM
    src_ds = ogr.Open(pf)
    src_layer = src_ds.GetLayer()

