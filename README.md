# Description
This codebase generates all the simulations and data presented in our paper [1]. Briefly, a single cell is modeled energetically within the phase-field framework and migrates on two types of confinement geometries: two-state and rectangular. Our main goal is to investigate whether one model for the cell is sufficient at capturing the rich spectrum of distinct migratory behaviors observed experimentally [2].

[1] Pedrom Zadeh, & Brian A. Camley. (2024). Nonlinear dynamics of confined cell migration – modeling and inference.

[2] Brückner, D.B., Fink, A., Schreiber, C. et al. Stochastic nonlinear dynamics of confined cell migration in two-state systems. Nat. Phys. 15, 595–601 (2019).


# Installation Guide
1. Clone the repo:
    ```bash
    git clone https://github.com/pedromzadeh/hopper.git
    cd hopper
    ```

2. From project's root, create the necessary `conda` environment:
   - Install `conda` if you don't have it.
   - Update `conda` to make sure you have the latest version.
   - Now proceed to create an environment from the file:
    ```bash
    conda env create -f environment.yaml
    conda activate hopper
    ```

3. Install local packages at the root of repo:
    ```bash
    pip install -e .
    ```

# Configuration Files
Cells are initialized from `cell.yaml` files with the following fields:

| Parameter          |  Type   | Description                                                   |
| :----------------- | :-----: | :------------------------------------------------------------ |
| `id`               |  `int`  | The cell ID                                                   |
| `R_eq`             | `float` | Specifies the target area of the cell                         |
| `R_init`           | `float` | Specifies the initial area of the cell                        |
| `R_ten_factor`     | `float` | Specifies the perimeter at which global tension is activated  |
| `lam`              | `float` | Sets phase field interfacial thickness                        |
| `polarity_mode`    |  `str`  | Set to IM                                                     |
| `gamma`            | `float` | Strength of cell line tension                                 |
| `eta`              | `float` | The coefficient of friction                                   |
| `N_wetting`        |  `int`  | Time to push the cell onto substrate to facilitate wetting    |
| `alpha`            | `float` | Strength of motility force                                    |
| `mu_mvg`           | `float` | Mean magnitude of MVG patch                                   |
| `sigma_mvg`        | `float` | Std of magnitude of MVG patch                                 |
| `tau`              | `float` | Decay timescale of cell polarity                              |
| `tau_x`            | `float` | Decay timescale of cell-ECM interaction                       |
| `tau_ten`          | `float` | Decay timescale of global tension degrading polarity          |
| `tau_mvg`          | `float` | Time interval for adding MVG patches                          |
| `interpolate_cntr` | `bool`  | Whether cell contour points are interpolated                  |
| `perturbation`     |  `int`  | 0: default polarity model; 1: turns off filopodia probability |
| `nu`               | `float` | Set to 0 (deprecated)                                         |

Confinements are initialized from `simbox.yaml` files with the following fields:
| Parameter              |  Type   | Description                                       |
| :--------------------- | :-----: | :------------------------------------------------ |
| `L_box`                |  `int`  | Size of the simulation box                        |
| `N`                    |  `int`  | Total number of simulation steps                  |
| `N_mesh`               |  `int`  | Number of simulation box lattice sites            |
| `dt`                   | `float` | Simulation time step                              |
| `simbox_view_freq`     |  `int`  | Interval at which snapshots are saved to file     |
| `stat_collection_freq` |  `int`  | Interval at which measurements are recorded       |
| `substrate.buffer`     |  `int`  | Set to 0                                          |
| `substrate.kind`       |  `str`  | Type of substrate: "two-state" or "rectangular"   |
| `substrate.sub_sep`    |  `int`  | Set to 73 for "two-state" and 0 for "rectangular" |
| `substrate.xi`         | `float` | Sets the phase-field thickness of the substrate   |

The configuration files for all the simulation parameters referenced in the paper are in `_server/sim_data/configs`. If you prefer to define your own configuration file, then you *must* specify all the parameters listed in the table above in `cell.yaml` and `simbox.yaml` files respectively and abide by the tree structure below:

```
hopper/
  └─── configs/IM/
         └─── grid_id[placehoder for an integer]
               └─── cell.yaml
               └─── simbox.yaml  
```

You can leverage the existing files in `configs/IM` to generate a few core results shown in our paper [1]:
- `grid_id2`: Figure 3(c) -- default cell with smaller radius in two-state geometry, which exhibits bistability.
- `grid_id10`: Figure 3(a) -- default cell in two-state geometry, which exhibits limit cycles.
- `grid_id12`: Figure 8 -- default cell with $\tau_\chi \to \infty$.
- `grid_id26`: Figure 3(b) -- default cell in rectangular geometry, which exhibits stationary behavior.
- `grid_id151` Default cell moving in a free 2D environment
The remaining configuration files in that directory are for other perturbations, such as those seen in Figs. 4, 6, and S4. Consult `cell.yaml` to see the cell specifications, and `simbox.yaml` to see the confining geometry.

# Running a Simulation
As an end user, you really need to only interact with the `_server/single_run.py` file, which executes one full simulation for a particular cell.

Simply run 
```bash 
cd _server
python single_run.py arg   # arg: integer representing the grid ID housed in /configs/IM
```
For example, `python single_run.py 2` will run a full-scale simulation for the cell defined according to `configs/IM/grid_id2/*`. Simulations should finish in about 15-20 mintues.

The results are stored in `output/IM/grid_id2/run_2/results.csv`, and the following important values are recorded:

1. the position of the cell $(x, y)$, 
2. time, in hours, of the measurement $t$.

# Processing a Simulation Run
The `analysis` package defines all the post-simulation processing we need. It acts on the raw position trajectory to compute velocity and acceleration time series. It then creates an acceleration field in the xv- phase-space. We compute the acceleration footprint by averaging this field over thousands of simulations. Play with the notebook `notebooks/ray_tracing.ipynb` to see the various stages of post-simulation processing.

> [!WARNING] 
> You must run the Jupyter notebook in the `conda` environment we created above since it needs the `analysis` package. See this [stackoverflow thread](https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook) for further instructions.

## Disclaimer
This codebase was written for scientific purposes, and as such, it was not implemented with scalability or generality in mind, and there is no continued support or updates for it. If you wish to fork this repository and use it in your projects, you may contact me for individual support at [zadeh@jhu.edu](mail:zadeh@jhu.edu).
