from typing import Union
from pathlib import Path
import yaml, json


def load_dict(file: Union[str, Path]):
    path = Path(file) if isinstance(file, str) else file
    with open(file, 'r') as inf:
        if path.suffix in ['.yml', '.yaml']:
            data = yaml.safe_load(inf)
        elif path.suffix in ['.json']:
            data = json.load(inf)
        else:
            raise ValueError(f'Wrong dictionary type: "{path.suffix}"')
    return data

