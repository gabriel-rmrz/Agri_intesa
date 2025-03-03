import argparse

def parse_command_line(argv, is_local=False):
  file_path='configs/default_parameters.yaml'
  if is_local:
    file_path='configs/default_parameters_localDB.yaml'
  parser = argparse.ArgumentParser(description="Retrieval of images using a wms")
  parser.add_argument('-c','--config_file', help="Path to the configuration file in yaml format", nargs='?', default=file_path, type=str, required=False)
  return vars(parser.parse_args(argv))
