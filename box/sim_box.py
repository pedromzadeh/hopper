import yaml


class SimulationBox:
    def __init__(self, config_file):
        """
        Initialize the simulation box.

        Parameters
        ----------
        config_file : str
            Path to configuration file.
            Expected to include:
                - N : total simulation time
                - dt : timestep
                - N_mesh : number of lattice sites
                - L_box : physical size of the box.
                - stat_collection_freq: frequency of data collection
                - simbox_view_freq: frequency of plotting
                - substrate: dict with keys "xi", "kind", "buffer".

            The resolution of the simulation is dx = L_box / (N_mesh - 1).
        """
        # read simulation box parameters
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        self.sim_time = config["N"]
        self.dt = config["dt"]
        self.N_mesh = config["N_mesh"]
        self.L_box = config["L_box"]
        self.n_stats = config["stat_collection_freq"]
        self.n_view = config["simbox_view_freq"]
        self.dx = self.L_box / (self.N_mesh - 1)
        self.sub_config = config["substrate"]
