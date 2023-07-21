from helper_functions import helper_functions as hf
import numpy as np
import yaml


class Cell:
    """
    The Cell class defines a phase-field cell and implements methods to update
    the cell's field from one timestep to the next.

    Attributes
    ----------
    self.id : int
        Specifies the ID (by index) of the cell.

    self.R_eq : float
        Specifies the target radius, thus target area of the cell.

    self.R_init : float
        Specifies the initial radius for building the cell.

    self.N_wetting : int
        Total time cell accelerates down to wet on substrate.

    self.center : list of floats
        Specifies the centroid of the cell when building it.

    self.gamma : float
        Specifies cell surface tension.

    self.A : float
        Specifies cell-substrate strength of adhesion.

    self.g : float
        Specifies cell-substrate strength of repulsion.

    self.beta : float
        Specifies how strongly shape changes affect the polarization field.

    self.mag_mean : float
        Specifies the mean magnitude of the MVG patch for polarization field.

    self.mag_std : float
        Specifies the variance of magnitude of the MVG patch for polarization field.

    self.alpha : float
        Specifies the magnitude of motility force.

    self.tau : float
        Specifies the decay timescale of the polarization field.

    self.tau_x : float
        Specifies the decay timescale of the polarization field
        due to interaction with the micro-pattern (mp).

    self.tau_ten : float
        Specifies the decay timescale of the polarization field
        due to shape elongations.

    self.lam : float
        Specifies the phase field interfacial thickness.

    self.eta : float
        Cell's friction coefficient.

    self.nu : float
        Specifies the strength with which polarization target is maintained.

    self.polarity_mode : str
        Specifies the modality used to update the cell's polarity.

    self.phi : ndarray of shape (N_mesh, N_mesh)
        The phase-field of the cell.

    self.W : ndarray of shape (N_mesh, N_mesh)
        Cumulative substrate the cell sees and interacts with.

    self.contour : list of ndarray, of shape (s, m, 2)
        The (y, x) points defining the 1/2-contour of the field, where m is the
        number of such points and s is the number of closed contours.

    self.cm : ndarray of shape (2, 2)
        Rows are previous and current CM, respectively, with ordering (x, y).

    self.p_field : ndarray of shape (N_mesh, N_mesh)
        The cell's polarization field, which models cytoskeletal actin dynamics.
        This is rho * phi.

    self.v_cm : ndarray of shape (2,)
        Cell center of mass velocity (v_cm_x, v_cm_y).

    self.vx, self.vy : ndarray of shape (N_mesh, N_mesh)
        Specify the velocity fields.

    self.simbox : SimulationBox object
        Directly gives each cell access to simulation box parameters.

    self.rng : np.random.RandomState
        Specifies an instance of random bit generator local to this Cell.
        All calls for random numbers should be made from this generator.

    Methods
    -------
    __init__(self, config_file)
        Initializes the cell.

    create(self, R, center)
        Builds the phase field.

    _tahn(self, r, R, epsilon)
        Returns the 2D hyperbolic tangent.

    _load_parameters(self, path)
        Loads cell's hyperparameters from file.
    """

    def __init__(self, config_file, _sim_box_obj, seed):
        """
        Initializes the cell object with some hypterparameters and physical and
        spatial features.

        Parameters
        ----------
        config_file : str
            Path to where hyperparameters of the cell are stored.

        _sim_box_obj : SimulationBox object
            Gives access to simulation box parameters directly to each cell.

        seed : int
            Seeds the local RandomState generator.
        """
        # read cell hyperparameters from file
        self.simbox = _sim_box_obj
        self._load_parameters(config_file)

        # random number generator local to this instance of Cell
        self.rng = np.random.default_rng(seed=seed)

        # spatial features of the cell
        self._init_center()
        self._init_phase_field()
        self._init_mvg_generator()

        self.contour = hf.find_contour(
            self.phi, interpolate=self.pol_model_kwargs["interpolate_cntrs"]
        )
        self.cm = np.array([self.center, self.center])
        self.p_field = self.rng.uniform(0, 1, size=self.phi.shape) * self.phi

        # physical features of the cell
        self.vx = np.zeros((_sim_box_obj.N_mesh, _sim_box_obj.N_mesh))
        self.vy = np.zeros((_sim_box_obj.N_mesh, _sim_box_obj.N_mesh))
        self.v_cm = np.array([0, 0])

    def _init_phase_field(self):
        """
        Computes the cell's phase-field from intial values and sets self.phi.
        """
        N_mesh, dx = self.simbox.N_mesh, self.simbox.dx
        _phi = np.zeros((N_mesh, N_mesh))
        center, R = self.center, self.R_init
        epsilon = self.lam
        one_dim = np.arange(N_mesh)
        x, y = np.meshgrid(one_dim, one_dim)
        r = np.sqrt((center[1] - y * dx) ** 2 + (center[0] - x * dx) ** 2)
        _phi[y, x] = self._tanh(r, R, epsilon)
        self.phi = _phi

    def _tanh(self, r, R, epsilon):
        return 1 / 2 + 1 / 2 * np.tanh(-(r - R) / epsilon)

    def _init_center(self):
        def _two_state_init(d):
            x1 = self.simbox.L_box / 2 - d
            x2 = self.simbox.L_box / 2 + d
            centers = [[x1, 25], [x2, 25]]
            return centers[self.rng.integers(0, 2)]

        def _rectangular_init(d):
            x1 = self.simbox.L_box / 2 - d
            x2 = self.simbox.L_box / 2 + d
            return [self.rng.uniform(x1, x2), 25]

        # config is in microns -->
        # / 6 for PF units; / 2 for symmetry
        d = self.simbox.sub_config["sub_sep"] / (2 * 6)
        kind = self.simbox.sub_config["kind"]
        self.center = (
            _two_state_init(d) if kind == "two-state" else _rectangular_init(d)
        )

    def _init_mvg_generator(self):
        from polarity.mvgaussian import MVGaussian

        d = np.arange(self.simbox.N_mesh)
        x, y = np.meshgrid(d, d)
        X = np.array(list(zip(x.flatten(), y.flatten())))
        self.mvg_gen = MVGaussian(X)

    def _load_parameters(self, path):
        def _polarity_params(config):
            return {
                "add_rate": config["add_rate"],
                "mag_mean": config["mag_mean"],
                "mag_std": config["mag_std"],
                "tau": config["tau"],
                "tau_x": config["tau_x"],
                "tau_ten": config["tau_ten"],
                "interpolate_cntrs": config["interpolate_cntrs"],
                "R_ten_factor": config["R_ten_factor"],
                "perturbation": config["perturbation"],
            }

        with open(path, "r") as file:
            config = yaml.safe_load(file)

        self.id = config["id"]
        self.R_eq = config["R_eq"]
        self.R_init = config["R_init"]
        self.gamma = config["gamma"]
        self.nu = config["nu"]
        self.alpha = config["alpha"]
        self.lam = config["lam"]
        self.polarity_mode = config["polarity_mode"]
        self.N_wetting = config["N_wetting"]
        self.eta = config["eta"]
        self.pol_model_kwargs = _polarity_params(config)
