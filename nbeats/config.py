# config.py
import argparse
from dataclasses import dataclass
from pathlib import Path
import yaml
#TODO: create NBeatsConfig dataclass
@dataclass
class NBeatsConfig:
  stack_layers: int
  block_layers: int = 4
  fc_units: int = 2048
  fc_layers: int = 4
  trend_order: int = 3
# We only specify the yaml file from argparse and handle rest
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-f", "--config_file", default="configs/default.yaml", help="Configuration file to load.")
ARGS = parser.parse_args()

# Let's load the yaml file here
with open(ARGS.config_file, 'r') as f:
  config = yaml.load(f)
print(f"Loaded configuration file {ARGS.config_file}")

def extern(func):
  """Wraps keyword arguments from configuration."""
  def wrapper(*args, **kwargs):
    """Injects configuration keywords."""
    # We get the file name in which the function is defined, ex: train.py
    fname = Path(func.__globals__['__file__']).name
    # Then we extract arguments corresponding to the function name
    # ex: train.py -> load_data
    conf = config[fname][func.__name__]
    # And update the keyword arguments with any specified arguments
    # if it isn't specified then the default values still hold
    conf.update(kwargs)
    return func(*args, **conf)
  return wrapper
