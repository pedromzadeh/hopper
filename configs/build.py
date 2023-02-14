import numpy as np
import yaml
import os

from sklearn.model_selection import ParameterGrid

pol_type = "IM"
N_cells = 1

# parameters to sweep through
#   - beta  : strength with which pol field responds to shape changes
#   - tau   : timescale of pol field decay
#   - D     : magnitude of noise

betas = np.linspace(1.2, 1.6, 2)
gammas = np.linspace(1.0, 1.6, 2)
Ds = np.linspace(0.02, 0.1, 2)

default_dict = {
    # "gamma": [1.2],
    "A": [0],
    "R_eq": [2.7],
    "R_init": [2.7],
    "eta": [0.5],
    "g": [0],
    "lam": [0.8],
    "N_wetting": [500],
    "alpha": [80],
    "tau": [5],
    "tau_mp": [0.05],
    "id": [0],
    "polarity_mode": [str(pol_type).upper()],
}

param_grid = {
    "beta": list(map(float, betas)),
    "gamma": list(map(float, gammas)),
    "D": list(map(float, Ds)),
} | default_dict

grid = list(ParameterGrid(param_grid))

print(f"betas: {betas} \ngammas: {gammas}\nDs: {Ds}\n")
print(f"Total # of configs: {len(grid)}")

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{pol_type}")

for id, params in enumerate(grid):
    path = os.path.join(root, f"grid_id{id}")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, f"cell{0}.yaml"), "w") as yfile:
        yaml.dump(params, yfile)
