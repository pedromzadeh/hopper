import numpy as np


def dFpol_dP(cell):
    P_tot_target = 10
    P_tot = np.sum(cell.phi * cell.p_field) * cell.simbox.dx**2
    nu = 10
    return 2 * nu * (1 - P_tot / P_tot_target) * (-1 / P_tot_target)


def adaptive_pol_field(cell, dphi_dt, mp):
    """
    Returns the change in polarization field experienced in one timestep.

    Parameters
    ----------
    cell : Cell object
        The cell whose polarization field is to update.

    dphi_dt : np.ndarray of shape (N_mesh, N_mesh)
        The change in cell shape experienced during this timestep.

    Returns
    -------
    np.ndarray of shape (N_mesh, N_mesh)
        The change in the field.
    """
    p_field = cell.p_field
    phi = cell.phi
    dt = cell.simbox.dt
    dx = cell.simbox.dx

    # parameters relevant to field update
    D = cell.D
    beta = cell.beta
    tau = cell.tau  # x8 to get minutes
    tau_x = cell.tau_mp

    noise = np.sqrt(4 * D**2 * dt / dx**2) * cell.rng.randn(*phi.shape)
    return (
        (dt * phi * beta * dphi_dt)
        - (dt / tau * p_field)
        + (noise * phi)
        - (dt / tau_x * mp * phi)
        - (dt * dFpol_dP(cell))
    )
