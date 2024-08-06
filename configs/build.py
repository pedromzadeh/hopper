import ruamel.yaml as yaml
import os
from shutil import copyfile


def build(dimensions, config_id):
    yaml_obj = yaml.YAML()
    yaml_obj.indent(sequence=4, offset=2)
    simbox_yaml = yaml_obj.load(open("configs/base_simbox.yaml"))

    for key, value in dimensions.items():
        simbox_yaml["substrate"][key] = value

    os.makedirs(f"configs/IM/grid_id{config_id}", exist_ok=True)
    copyfile("configs/base_cell.yaml", f"configs/IM/grid_id{config_id}/cell.yaml")
    yaml_obj.dump(simbox_yaml, open(f"configs/IM/grid_id{config_id}/simbox.yaml", "w"))


if __name__ == "__main__":
    dims = [
        [[38, 38], [38, 38]],
        [[38, 38], [28, 28]],
        [[38, 38], [32, 32]],
        [[42, 42], [28, 28]],
        [[42, 42], [32, 32]],
        # rectangle basins
        [[25, 48], [25, 48]],
        [[25, 48], [48, 25]],
        [[48, 25], [25, 48]],
    ]

    for id, basin_dims in enumerate(dims):
        build({"basin_dims": basin_dims, "bridge_dim": [16, 10]}, id)
