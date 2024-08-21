import ruamel.yaml as yaml
import os
from shutil import copyfile


def build(dimensions, config_id):
    yaml_obj = yaml.YAML()
    yaml_obj.indent(sequence=4, offset=2)
    simbox_yaml = yaml_obj.load(open("configs/base_simbox.yaml"))

    simbox_yaml["simbox_view_freq"] = int(1e10)

    for key, value in dimensions.items():
        simbox_yaml["substrate"][key] = value

    os.makedirs(f"configs/IM/grid_id{config_id}", exist_ok=True)
    copyfile("configs/base_cell.yaml", f"configs/IM/grid_id{config_id}/cell.yaml")
    yaml_obj.dump(simbox_yaml, open(f"configs/IM/grid_id{config_id}/simbox.yaml", "w"))


if __name__ == "__main__":
    dims = [
        # square basins
        # 38: default
        # 28 -> 30
        # 32 -> 34
        [[38, 38], [38, 38]],
        [[38, 38], [30, 30]],
        [[38, 38], [34, 34]],
        [[42, 42], [30, 30]],
        [[42, 42], [34, 34]],
        # rectangle basins, scaled up by a factor of 1.2
        [[30, 58], [30, 58]],
        [[30, 58], [58, 30]],
        # [[58, 30], [30, 58]],
    ]

    for id, basin_dims in enumerate(dims):
        build({"basin_dims": basin_dims, "bridge_dim": [35, 17]}, id)
