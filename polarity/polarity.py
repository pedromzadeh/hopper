import numpy as np
from scipy.interpolate import RegularGridInterpolator


def polarity_at_cntr(cell, method="linear"):
    p_field = np.where(cell.p_field > 0, cell.p_field, 0)
    x = np.arange(cell.simbox.N_mesh)
    interp = RegularGridInterpolator((x, x), p_field, method=method)
    cntr = cell.contour[0]
    return interp(cntr)


def shape_induced_stress(cell, perim_0):
    """
    Returns an estimate for the stress caused by elongated shapes.

    Parameters
    ----------
    cell : Cell object
        The cell whose polarity field we're updating.

    perim_0 : float
        Perimeter beyond which membrane tension kicks in.

    Returns
    -------
    float
        Scalar value representing cell membrane tension.
    """
    cntr = cell.contour[0][:, ::-1] * cell.simbox.dx
    perim = np.sqrt(np.sum(np.diff(cntr, axis=0) ** 2, axis=1)).sum()
    return (perim / perim_0) ** 2 if perim > perim_0 else 0


def dFpol_dP(cell, nu, target):
    """
    Returns change in cell polarity so to keep total polarization constant.

    Parameters
    ----------
    cell : Cell object
        The cell whose polarization field is in question.

    nu : float
        Specifies the strength of this term. 0 turns it off.

    target : float
        Specifies the constant target polarization amount.

    Returns
    -------
    float
        The scalar change in polarization field.
    """
    P_tot = np.sum(cell.phi * cell.p_field) * cell.simbox.dx**2
    return 2 * nu * (1 - P_tot / target) * (-1 / target)


def cntr_probs_filopodia(cell, grad_phi, mp, delta_l=4, method="linear", norm=True):
    """
    Probability of each contour point assigned based on the existence of
    micropattern a given normal distance away from the point.

    Parameters
    ----------
    cell : Cell object
        The cell whose contour points we're after.

    grad_phi : np.ndarray of shape (2, N_mesh, N_mesh)
        Gradient of the phase-field, with grad_x, grad_y, respectively.

    mp : np.ndarray of shape (N_mesh, N_mesh)
        The field for the micropatntern.

    delta_l : float, optional
        Amount by which we protrude outside the cell, in phase-field units.
        Default is 4.

    method : str
        One of "linear", "nearest", "slinear", "cubic", "quintic", "pchip"
        for interpolated values.
        By default "linear", which interpolates.

    norm : bool, optional
        If True, probabilities are normalized, else not, by default True

    Returns
    -------
    np.ndarray of shape (# of contour points, )
    """
    cntr = cell.contour[0][:, ::-1]
    grad_phi_norm = np.sqrt(np.sum(grad_phi * grad_phi, axis=0)) + 1e-10
    n_hat = -grad_phi / grad_phi_norm
    cntr_n_hats = np.array(
        [[n_hat[0][int(y), int(x)], n_hat[1][int(y), int(x)]] for x, y in cntr]
    )
    cntr_shifted = np.array(
        [
            [x + delta_l * nx, y + delta_l * ny]
            for x, y, nx, ny in np.hstack((cntr, cntr_n_hats))
        ]
    )

    x = np.arange(cell.simbox.N_mesh)
    interp = RegularGridInterpolator((x, x), 1 - mp, method=method)
    p = interp(cntr_shifted[:, ::-1])
    if norm:
        return p / p.sum()
    return p


def cntr_probs_feedback_times_radius(cell, norm=True):
    # probs due to feedback and distance from cell CM
    cntr = cell.contour[0][:, ::-1]
    in_frame_cntr = cntr * cell.simbox.dx - cell.cm[1]
    p_at_cntr = polarity_at_cntr(cell)
    radii = np.linalg.norm(in_frame_cntr, axis=1)
    p = p_at_cntr * radii + 1e-5

    if norm and p.sum() > 0:
        return p / p.sum()
    return p


def cntr_probs_feedback(cell, norm=True):
    """
    Probability of each contour point assigned based on the existing
    polarity value at that point.

    Parameters
    ----------
    cell : Cell object
        The cell whose contour points we're after.

    norm : bool, optional
        If True, probabilities are normalized, else not, by default True

    Returns
    -------
    np.ndarray of shape (# of contour points, )
    """
    p_at_cntr = polarity_at_cntr(cell)
    p = p_at_cntr + 1e-5

    if norm and p.sum() > 0:
        return p / p.sum()
    return p


def mvg_patch(cell, cntr_probs=None, cov_ii=20, cov_ij=0, cntr_pt=None, return_c=False):
    """
    Returns a multivariate Gaussian (mvg) centered at a contour point, which is
    either passed as input or drawn from the input probability mass distribution.

    Parameters
    ----------
    cell : Cell object
        Cell whose contour we're interested in.

    cntr_probs : np.ndarray of shape (# of contour points), optional
        Normalized probability mass for contour points
        experiencing a protrusive mvg patch, by default None

    cov_ii : float, optional
        Diagonal element of the covariance matrix, by default 20

    cov_ij : float, optional
        Off-diagonal element of the covariance matrix, by default 0

    cntr_pt : list or np.ndarray of shape (2, ), optional
        Specific contour point (x, y) about which an mvg patch is created,
        by default None

    Returns
    -------
    np.ndarray of shape (N_mesh, N_mesh)
        A multivariate Gaussian defined in the support space about a contour point.
    """

    def _pick_c():
        """
        Returns a contour point, format (x, y).
        """
        if cntr_pt is not None:
            return cntr_pt

        cntr = cell.contour[0][:, ::-1]
        return cntr[cell.rng.choice(range(len(cntr)), p=cntr_probs)]

    c = _pick_c()
    cov = np.array([[cov_ii, cov_ij], [cov_ij, cov_ii]])
    N_mesh = cell.simbox.N_mesh

    if return_c:
        return cell.mvg_gen.pdf(c, cov).reshape(N_mesh, N_mesh), c
    return cell.mvg_gen.pdf(c, cov).reshape(N_mesh, N_mesh)


def pixel_random_noise(cell, D):
    dt = cell.simbox.dt
    dx = cell.simbox.dx
    phi = cell.phi

    noise = np.sqrt(4 * D**2 * dt / dx**2) * cell.rng.randn(*phi.shape)
    return noise * phi


def update_field(cell, grad_phi, mp, n):
    """
    Returns the updated polarity field.

    Parameters
    ----------
    cell : Cell object
        The cell whose polarity field we're updating.

    grad_phi : np.ndarray of shape (2, N_mesh, N_mesh)
        Gradient of the phase-field, with grad_x, grad_y, respectively.

    mp : np.ndarray of shape (N_mesh, N_mesh)
        The field for the micropatntern.

    mvg_patch : np.ndarray of shape (N_mesh, N_mesh)
        A multivariate Gaussian acting as a protrusive impulse.

    model_args : dict
        Relevant model parameters -- tau, tau_x, tau_ten, tau_mvg, perim_0

    Returns
    -------
    np.ndarray of shape (N_mesh, N_mesh)
        Updated cell polarity field.
    """

    # unpack
    p_field = cell.p_field
    phi = cell.phi
    dt = cell.simbox.dt

    R_ten = cell.pol_model_kwargs["R_ten_factor"] * cell.R_eq
    tau = cell.pol_model_kwargs["tau"]
    tau_x = cell.pol_model_kwargs["tau_x"]
    tau_ten = cell.pol_model_kwargs["tau_ten"]
    tau_mvg = cell.pol_model_kwargs["tau_mvg"]
    mu_mvg = cell.pol_model_kwargs["mu_mvg"]
    sigma_mvg = cell.pol_model_kwargs["sigma_mvg"]

    # PMF for contours to see MVG -- given perturbation ID
    cntr_probs = _model_perturbation_probs(
        cell.pol_model_kwargs["perturbation"], cell, grad_phi, mp
    )

    # add MVG patch every tau_mvg phase-field units of time
    patch = 0
    if n % (tau_mvg // dt) == 0:
        mag = cell.rng.normal(loc=mu_mvg, scale=sigma_mvg)
        patch = mag * mvg_patch(cell, cntr_probs)

    # assess membrane tension
    target_perimeter = 2 * np.pi * R_ten
    mem_tension = shape_induced_stress(cell, target_perimeter)

    # update polarity field
    return p_field + (
        (patch * phi)
        - (dt / tau * p_field)
        - (dt / tau_x * mp * phi)
        - (dt / tau_ten * mem_tension)
    )


def _model_perturbation_probs(pert_id, cell, grad_phi, mp):
    if pert_id == 0:
        p1 = cntr_probs_filopodia(cell, grad_phi, mp)
        p2 = cntr_probs_feedback(cell)
        cntr_probs = p1 * p2
        return cntr_probs / cntr_probs.sum()

    elif pert_id == 1:
        cntr_probs = cntr_probs_feedback(cell)
        return cntr_probs / cntr_probs.sum()

    else:
        raise ValueError(f"{pert_id} not understood.")
