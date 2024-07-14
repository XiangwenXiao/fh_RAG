from datetime import datetime
from pathlib import Path
from string import Template
from typing import List, Union

import yaml
def read_yaml(yaml_path: Union[str, Path]):
    with open(str(yaml_path), "rb") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data