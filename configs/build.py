import numpy as np
import yaml
import os
import sys

from sklearn.model_selection import ParameterGrid


def simbox_configs(snapshots, L_two_state=73):
    if snapshots.lower() == "many":
        n_simbox_view = 1000
    elif snapshots.lower() == "few":
        n_simbox_view = 500000
    else:
        raise ValueError(f"{snapshots} must be either `many` or `few`.")

    base_config = {
        "N": 240000,  # total simulation time
        "dt": 0.00075,  # time step (~ 0.36s given 1pft = 8min)
        "N_mesh": 200,  # lattice grid size
        "L_box": 50,  # size of simulation box in real units
        "stat_collection_freq": 500,  # frequency of collecting pre-collision stats
        "simbox_view_freq": n_simbox_view,  # frequency of outputting the view of simbox
    }
    two_state = {
        "substrate": {
            "xi": 0.2,
            "kind": "two-state",
            "buffer": 0,
            "sub_sep": L_two_state,
        }
    }
    single_state = {
        "substrate": {
            "xi": 0.2,
            "kind": "rectangular",
            "buffer": 0,
            "sub_sep": L_two_state,
        }
    }

    return (base_config | two_state, base_config | single_state)


def cell_configs(pol_type, grid_axes_kwargs):
    # static cell parameters
    cell_kwargs = {
        "N_wetting": [500],
        "R_init": [2.75],
        "eta": [0.5],
        "nu": [0],
        "lam": [0.8],
        "alpha": [50],
        "id": [0],
        "polarity_mode": [str(pol_type).upper()],
    }

    # static polarity model params
    # perturbation ids:
    #       0) None
    #       1) P_feedback ~ P
    #       2) P_feedback ~ R
    #       3) P_filopodia ~ Uniform

    pol_model_kwargs = {
        "mu_mvg": [7.5],
        "tau": [0.5],
        "tau_x": [0.02],
        "tau_ten": [1.0],
        "interpolate_cntrs": [False],
        "R_ten_factor": [1.5],
        "perturbation": [1],
    }

    return list(ParameterGrid(grid_axes_kwargs | cell_kwargs | pol_model_kwargs))


if __name__ == "__main__":
    pol_type = "IM"
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{pol_type}")

    # grid search over...
    gammas = np.linspace(0.8, 1.8, 2)
    mag_stds = np.linspace(22.5, 37.5, 2)
    add_rates = np.linspace(0.0045, 0.009, 2)  # x 8 x 60 --> seconds
    R_eqs = np.linspace(2.5, 3.0, 2)

    grid_axes_kwargs = {
        "gamma": list(map(float, gammas)),
        "sigma_mvg": list(map(float, mag_stds)),
        "tau_mvg": list(map(float, add_rates)),
        "R_eq": list(map(float, R_eqs)),
    }
    grid = cell_configs(pol_type, grid_axes_kwargs)
    mps = simbox_configs(sys.argv[1])

    print(f"Writing {2*len(grid)} configuration files...")
    for k, sub_type in enumerate(["two_state", "single_state"]):
        for id, params in enumerate(grid):
            if k == 1:
                id += len(grid)

            path = os.path.join(root, f"grid_id{id}")
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

            # write simbox config
            with open(os.path.join(path, "simbox.yaml"), "w") as yfile:
                yaml.dump(mps[k], yfile)

            # write each cell config
            with open(os.path.join(path, "cell.yaml"), "w") as yfile:
                yaml.dump(params, yfile)
