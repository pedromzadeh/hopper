import numpy as np


def mvg_noise_patch(cell, mp, cov_ii, cov_ij):
    
    def pick_c(kind="mp-only"):
        from helper_functions import helper_functions as hf

        grad_x, grad_y, _ = hf.compute_gradients(cell.phi, cell.simbox.dx)
        grad_phi = np.array([grad_x, grad_y])
        grad_phi_norm = np.sqrt(np.sum(grad_phi * grad_phi, axis=0))
        n_hat = -grad_phi / grad_phi_norm

        cntr = cell.contour[0][:, ::-1]
        cntr_n_hats = np.array(
            [[n_hat[0][int(y), int(x)], n_hat[1][int(y), int(x)]] for x, y in cntr]
        )
        cntr_shifted = np.array(
            [
                [x + l * nx, y + l * ny]
                for x, y, nx, ny in np.hstack((cntr, cntr_n_hats))
            ]
        )

        l = 5
        p = 1 - mp
        cntr_probs = np.array([p[int(y), int(x)] for x, y in cntr_shifted])
        cntr_probs /= cntr_probs.sum()

        return cntr[np.random.choice(range(len(cntr)), p=cntr_probs)]

    # pick a contour point according to cntrs_probs
    c = pick_c()
    means = np.array(c)
    cov = np.array([[cov_ii, cov_ij], [cov_ij, cov_ii]])
    N_mesh = cell.simbox.N_mesh

    return cell.mvg_gen.pdf(means, cov).reshape(N_mesh, N_mesh)


def dFpol_dP(cell):
    P_tot_target = cell.P_target
    nu = cell.nu
    P_tot = np.sum(cell.phi * cell.p_field) * cell.simbox.dx**2
    return 2 * nu * (1 - P_tot / P_tot_target) * (-1 / P_tot_target)


def adaptive_pol_field(cell, dphi_dt, mp, args):
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

    D = cell.D
    tau_x, beta, tau_N, tau, sigma, tau_add, n = args

    if n % tau_add == 0:
        noise = sigma * mvg_noise_patch(cell, mp, cov_ii=20, cov_ij=0)
    else:
        noise = 0

    # noise = np.sqrt(4 * D**2 * dt / dx**2) * cell.rng.randn(*phi.shape)
    return (
        (dt * phi * beta * dphi_dt)
        - (dt / tau * p_field)
        + (dt / tau_N * noise * phi)
        - (dt / tau_x * mp * phi)
    )
