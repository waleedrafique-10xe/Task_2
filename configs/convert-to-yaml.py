"This is a utility script to convert JSON format configs to YAML"

import glob
import json
from pathlib import Path

import yaml

for file in glob.glob("./**.json"):
    print(file)
    path = Path(file)
    new_path = path.parent / (path.stem + ".yaml")

    with open(file, "r") as f:
        data = json.load(f)

    with open(new_path, "w") as f:
        f.write(yaml.dump(data, indent=4))
