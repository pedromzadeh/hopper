import numpy as np
import yaml
import os

from sklearn.model_selection import ParameterGrid

pol_type = "IM"
N_cells = 1

# parameters to sweep through
#   - p0    : threshold perimeter
#   - gamma : cell line tension

gammas = np.linspace(1.0, 1.0, 1)
p0s = np.linspace(2 * np.pi * 4.5, 2 * np.pi * 4.5, 1)

cell_params = {
    "A": [0],
    "N_wetting": [500],
    "R_eq": [3.0],
    "R_init": [3.0],
    "eta": [0.5],
    "g": [0],
    "lam": [0.8],
    "alpha": [50],
    "id": [0],
    "polarity_mode": [str(pol_type).upper()],
}

pol_model_args = {
    "tau_add_mvg": [5],
    "patch_mag": [1000],
    "tau": [0.5],
    "tau_x": [0.02],
    "tau_mvg": [0.1],
    "tau_ten": [0.1],
}

param_grid = (
    {
        "gamma": list(map(float, gammas)),
        "perim_0": list(map(float, p0s)),
    }
    | cell_params
    | pol_model_args
)

grid = list(ParameterGrid(param_grid))

print(f"Total # of configs: {len(grid)}")

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{pol_type}")

for id, params in enumerate(grid):
    path = os.path.join(root, f"grid_id{id}")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, f"cell{0}.yaml"), "w") as yfile:
        yaml.dump(params, yfile)
