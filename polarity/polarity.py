import numpy as np

from polarity.mvgaussian import MVGaussian


def mvg_noise_patch(cell, dphi_dt):

    cntrs = cell.contour[0]  # [y, x]
    probs = np.fabs(dphi_dt)
    cntrs_probs = np.array([probs[int(y), int(x)] for y, x in cntrs])
    cntrs_probs[np.isnan(cntrs_probs)] = 0
    cntrs_probs /= np.sum(cntrs_probs)

    # pick a contour point according to cntrs_probs
    c = cntrs[np.random.choice(range(len(cntrs)), p=cntrs_probs)][::-1]
    means = np.array(c)
    cov = np.array([[8, 4], [4, 8]])

    d = np.linspace(0, 200, 200)
    x, y = np.meshgrid(d, d)
    X = np.array(list(zip(x.flatten(), y.flatten())))

    MVG_f = MVGaussian(means, cov)

    return MVG_f.pdf(X).reshape(200, 200)


def dFpol_dP(cell):
    P_tot_target = cell.P_target
    nu = cell.nu
    P_tot = np.sum(cell.phi * cell.p_field) * cell.simbox.dx**2
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
