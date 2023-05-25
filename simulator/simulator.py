from box.sim_box import SimulationBox
from potential.force import Force
from cell.cell import Cell
from substrate.substrates import Substrate
from helper_functions import helper_functions as hf
from visuals.figure import Figure

import glob
import pandas as pd
import os


class Simulator:
    """
    Implements a simulator object, which runs two-body collisions and collects
    relevant statistics.

    Attributes
    ----------
    self.root_dir : str
        Path to project's root directory.

    Methods
    -------
    __init__(self, config_dir, results_dir, figures_dir)
        Initialize the simulator.

    execute(self, exec_id, save_results_at, save_figures_at)
        Run one complete simulation.

    _build_system(self, simbox)
        Builds the cell and substrate system.

    _define_paths(self)
        Setup various paths for the simulator to use.
    """

    def __init__(self):
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def execute(self, run_id, grid_id, polarity_type, seed):
        """
        Executes one complete simulation run. This is a child process spawned
        by multiprocessing.

        Parameters
        ----------
        run_id : int
            Defines the id of this particular run.

        grid_id : int
            Defines the id of the grid point in the 3D feature space.

        polarity_type : str
            Specifies the polarity type being used.

        seed : int or float
            Sets the seed for the random generator assigned to this run.
        """
        # define various paths
        paths = self._define_paths(run_id, grid_id, polarity_type)

        # initialize the simulation box
        simbox = SimulationBox(paths["simbox"])
        cell, chi = self._build_system(simbox, paths["cell"], cell_rng_seed=seed)

        # initialize the force calculator
        force_calculator = Force(paths["energy"])

        # collect cell positions
        cms = pd.DataFrame()

        # noise patch -- init MVG generator for the cell
        self._mvg_generator(cell)

        # carry out the simulation
        for n in range(simbox.sim_time):

            # collect statistics
            if n % simbox.n_stats == 0 and self._cell_whole(cell):
                time = n * cell.simbox.dt * 8 / 60  # time in hr
                cms = pd.concat(
                    [
                        cms,
                        pd.DataFrame(
                            [[*cell.cm[1], *cell.v_cm, time]],
                            columns=["x", "y", "vx", "vy", "time[hr]"],
                        ),
                    ]
                )

            # view the simulation box
            if n % simbox.n_view == 0:
                Figure.view_pol_field(
                    cell,
                    chi,
                    dpi=150,
                    path=os.path.join(paths["figures"], f"img_{n}.png"),
                )

            # update each cell to the next time step
            hf.evolve_cell(cell, force_calculator, chi, n)

            # if cell has escaped, end simulation
            if not self._cell_inside(cell, chi):
                exit("Cell has escaped the sim box. Exiting run.")

        # simulation is done; store data
        cms["D"] = cell.D
        cms["beta"] = cell.beta
        cms["tau"] = cell.tau
        cms["gamma"] = cell.gamma
        cms["Req"] = cell.R_eq
        cms.to_csv(paths["result"])

    def _build_system(self, simbox, cell_config, cell_rng_seed):
        """
        Builds the substrate and cell system.

        Parameters
        ----------
        simbox : SimulationBox object
            Defines the box.

        cell_config : str
            Path to directory containing cell's config.

        cell_rng_seed : int
            Sets the seed for local instance of Cell's bit generator.

        Returns
        -------
        tuple
            cell : Cell object
            chi : Substrate object

        Raises
        ------
        ValueError
            Ensures only 1 cell is defined.
        """
        # unpack
        N_mesh, L_box = simbox.N_mesh, simbox.L_box

        # define base substrate
        sub_config = simbox.sub_config
        xi = sub_config["xi"]
        kind = sub_config["kind"]
        # buffer = sub_config["buffer"]
        sub = Substrate(N_mesh, L_box, xi)
        if kind == "two-state":
            chi = sub.two_state_sub(bridge_width=15)
        elif kind == "rectangular":
            chi = sub.rectangular()
        else:
            raise ValueError(f"{kind} for substrate is not understood.")

        # read cell config files && ensure only 2 existr
        # IMPORTANT -- glob returns arbitrary order, sort
        config_file = sorted(glob.glob(os.path.join(cell_config, "cell*")))
        if len(config_file) != 1:
            raise ValueError("Ensure there is exactly 1 configuration file.")

        # initialize cells
        # set the cumulative substrate they will interact with
        cell = Cell(config_file[0], simbox, cell_rng_seed)
        cell.W = 0.5 * cell.g * chi

        return cell, chi

    def _define_paths(self, run_id, grid_id, polarity_type):
        ENERGY_CONFIG = os.path.join(self.root_dir, "configs/energy.yaml")

        SIMBOX_CONFIG = os.path.join(
            self.root_dir, f"configs/{polarity_type}/grid_id{grid_id}", "simbox.yaml"
        )

        CELL_CONFIG = os.path.join(
            self.root_dir, f"configs/{polarity_type}/grid_id{grid_id}", "cell.yaml"
        )

        run_root = os.path.join(
            self.root_dir,
            "output",
            f"{polarity_type}",
            f"grid_id{grid_id}",
            f"run_{run_id}",
        )
        if not os.path.exists(run_root):
            os.makedirs(run_root)

        RESULT_PATH = os.path.join(run_root, "result.csv")

        FIGURES_PATH = os.path.join(run_root, "visuals")
        if not os.path.exists(FIGURES_PATH):
            os.makedirs(FIGURES_PATH)

        return dict(
            simbox=SIMBOX_CONFIG,
            energy=ENERGY_CONFIG,
            cell=CELL_CONFIG,
            result=RESULT_PATH,
            figures=FIGURES_PATH,
        )

    def _cell_inside(self, cell, mp):
        cm = cell.cm[-1]
        x = int(cm[0] / cell.simbox.dx)
        y = int(cm[1] / cell.simbox.dx)
        return mp[y, x] < 0.5

    def _cell_whole(self, cell):
        return len(cell.contour) == 1

    def _mvg_generator(self, cell):
        import numpy as np
        from polarity.mvgaussian import MVGaussian

        N_mesh = cell.simbox.N_mesh
        d = np.linspace(0, N_mesh, N_mesh)
        x, y = np.meshgrid(d, d)
        X = np.array(list(zip(x.flatten(), y.flatten())))

        cell.mvg_gen = MVGaussian(X)
