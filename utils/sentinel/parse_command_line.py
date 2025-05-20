import argparse

def parse_command_line(argv):
  file_path='configs/default_parameters_sentinel.yaml'
  parser = argparse.ArgumentParser(description="Retrieval of images using SentinelHub")
  parser.add_argument('-c','--config_file', help="Path to the configuration file in yaml format", nargs='?', default=file_path, type=str, required=False)
  return vars(parser.parse_args(argv))
