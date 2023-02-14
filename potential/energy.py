import numpy as np
import yaml
from helper_functions import helper_functions as hf


class Energy:
    def __init__(self, config_file):
        """
        Initialize the energy object.

        Parameters
        ----------
        config_file : str
            Path to energy parameters config file.

        """
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        self.kappa = config["kappa"]
        self.omega = config["omega"]
        self.mu = config["mu"]

    def cahn_hilliard_energy(self, cell):
        """
        Computes the Cahn-Hilliard energy of the cell.

        Parameters
        ----------
        cell : Cell object
            The cell we want to compute this energy for.

        Returns
        -------
        float
        """
        grad_x, grad_y, _ = hf.compute_gradients(cell)
        lam, gamma, dx = cell.lam, cell.gamma, cell.simbox.dx

        grad_sqrd = grad_x**2 + grad_y**2
        integrand = 4 * cell.phi**2 * (1 - cell.phi) ** 2 + lam**2 * grad_sqrd
        return gamma / lam * np.sum(integrand) * dx**2

    def area_energy(self, cell):
        """
        Computes the energy cost incurred when cell area deviates from
        target area.

        Parameters
        ----------
        cell : Cell object
            The cell in quesion.

        Returns
        -------
        float
        """
        area = np.sum(cell.phi**2) * cell.simbox.dx**2
        area_target = np.pi * cell.R_eq**2
        return self.mu * (1 - area / area_target) ** 2

    def substrate_int_energy(self, cell, W):
        """
        Computes the interaction energy of the cell with the substrate.
        This consists of adhesion to the substrate's surface, which is
        proportional to 'A', and repulsion from its body, which is
        proportional to 'g'. Both these constants are properties of
        the substrate.

        Parameters
        ----------
        cell : Cell object
            The cell in question.

        W : ndarray of shape (N_mesh, N_mesh)
            Defines the cumulative substrate phase-field with which the
            cell interacts for both adhesion and repulsion.

        Returns
        -------
        float
        """
        res = np.sum(cell.phi**2 * (2 - cell.phi) ** 2 * W) * cell.simbox.dx**2
        return res

    def cell_adhesion_energy(self, cell, cells):
        """
        Computes the energy gain due to cell-cell adhesion.

        Parameters
        ----------
        cell : Cell object
            The cell of interest.

        cells : list of Cells
            All cells in the system.

        Returns
        -------
        float
        """
        j = 0 if cell.id == 1 else 1
        nc = cells[j]
        neighbors_present = nc.phi**2 * (1 - nc.phi) ** 2
        integrand = cell.phi**2 * (1 - cell.phi) ** 2 * neighbors_present
        return -self.omega / cell.lam * np.sum(integrand) * cell.simbox.dx**2

    def cell_repulsion_energy(self, cell, cells):
        """
        Computes the energy cost due to cell-cell overlap.

        Parameters
        ----------
        cell : Cell object
            The cell of interest.

        cells : list of Cells
            All cells in the system.

        Returns
        -------
        float
        """
        j = 0 if cell.id == 1 else 1
        nc = cells[j]
        neighbors_present = nc.phi**2
        integrand = cell.phi**2 * neighbors_present
        return self.kappa / cell.lam * np.sum(integrand) * cell.simbox.dx**2
