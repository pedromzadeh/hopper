from polarity import polarity
from skimage import measure
import numpy as np


def find_contour(field, level=0.5):
    """
    Computes the contour at the specified level.

    Parameters
    ----------
    field : ndarray of shape (n, n)
        Phase field.

    level : float, optional
        Level of the contour points, by default 0.5

    Returns
    -------
    list of ndarray of shape (m, 2)
        m is the number of points, and the ordering is (y, x). More than one
        element in the list suggests disjointed sets of contour points.
    """
    return measure.find_contours(field, level=level)


def compute_CM(cell):
    """
    Computes the center of mass of cell.

    Parameters
    ----------
    cell : Cell object
        Cell in question.

    Returns
    -------
    tuple of ndarray of shape (2,)
        current center-of-mass, new center-of-mass, respectively.
    """
    N_mesh, dx = cell.simbox.N_mesh, cell.simbox.dx
    x = np.arange(0, N_mesh, 1)
    xg, yg = np.meshgrid(x, x)
    cm_x, cm_y = np.sum(xg * cell.phi), np.sum(yg * cell.phi)
    norm = np.sum(cell.phi)

    return cell.cm[1], np.array([cm_x / norm * dx, cm_y / norm * dx])


def compute_v_CM(cell):
    """
    Computes the center of mass velocity.

    Parameters
    ----------
    cell : Cell object
        Cell in question.

    Returns
    -------
    float
    """
    delta_cm = np.diff(cell.cm, axis=0)[0]
    return delta_cm / cell.simbox.dt


def compute_area(cell, order=2):
    """
    Computes the approximate area of the cell as $int phi^(order) dr$.

    Parameters
    ----------
    cell : Cell object
        Cell in question.

    order : int, optional
        Specifies the power of phi. Larger values give smaller areas as field
        is tightened, by default 2

    Returns
    -------
    float
    """
    return np.sum(np.power(cell.phi, order) * cell.simbox.dx**2)


def compute_gradients(field, dx):
    """
    Computes grad(field) and laplacian(field).

    Parameters
    ----------
    field : ndarray of shape (n, n)
        Phase field.

    dx : float
        Resolution of the simulation box.

    Returns
    -------
    tuple of ndarray of shape (field.shape[0], field.shape[0])
            grad_x, grad_y, laplacian
    """
    # compute the gradients, each component is separate
    field_right = np.roll(field, 1, axis=1)
    field_left = np.roll(field, -1, axis=1)
    field_up = np.roll(field, -1, axis=0)
    field_down = np.roll(field, 1, axis=0)

    # compute the gradients, each component is separate
    grad_x = (field_left - field_right) / (2 * dx)
    grad_y = (field_up - field_down) / (2 * dx)

    # compute laplacian
    laplacian = (field_left + field_right + field_up + field_down - 4 * field) / dx**2

    return (grad_x, grad_y, laplacian)


def evolve_cell(cell, force, mp, n):
    """
    Evolves the cell by updating its class variables from time t to time t + dt.
    Attributes updated are
    1. field
    2. polarization field == rho * phi
    3. center of mass
    4. center of mass speed
    5. velocity fields, v_x and v_y
    6. contour of the field.

    Parameters
    ----------
    cell : Cell object
        Cell of interest to update.

    force : Force object
        Gives access to computing forces.

    mp : Substrate object
        Specifies the micropattern.

    n : int
        Simulation timestep. Needed for deciding when to add MVG patches.
    """

    # needed more than once
    grad_x, grad_y, _ = compute_gradients(cell.phi, cell.simbox.dx)
    grad_phi = np.array([grad_x, grad_y])
    eta = cell.eta
    phi = cell.phi
    dt = cell.simbox.dt

    # contour PMF to add mvg patch
    p1 = polarity.cntr_probs_filopodia(cell, grad_phi, mp)
    p2 = polarity.cntr_probs_feedback(cell, grad_phi)
    cntr_probs = p1 * p2
    cntr_probs /= cntr_probs.sum()

    # add MVG patch according to a Poisson process
    tau_add = _poisson_add_time(cell.rng, cell.pol_model_kwargs["add_rate"])
    if n % tau_add == 0:
        mag = cell.rng.normal(
            loc=cell.pol_model_kwargs["mag_mean"], scale=cell.pol_model_kwargs["mag_std"]
        )
        mvg_patch = mag * polarity.mvg_patch(cell, cntr_probs)
    else:
        mvg_patch = 0

    # phi_(n+1)
    phi_i_next, dF_dphi = _update_field(cell, grad_phi, force)
    dphi_dt = (phi_i_next - phi) / dt

    # polarization field (n+1)
    p_field_next = polarity.update_field(cell, mp, mvg_patch, cell.pol_model_kwargs)

    # compute motility forces at time n
    fx_motil, fy_motil = force.cyto_motility_force(cell, grad_phi, mp)

    # compute thermodynamic forces at time n
    fx_thermo = dF_dphi * grad_x
    fy_thermo = dF_dphi * grad_y

    # UPDATE class variables now
    cell.phi = phi_i_next
    cell.p_field = p_field_next
    cell.contour = find_contour(cell.phi)
    cell.cm = compute_CM(cell)
    cell.v_cm = compute_v_CM(cell)
    cell.vx = (fx_thermo + fx_motil) / eta
    cell.vy = (fy_thermo + fy_motil) / eta


def _update_field(cell, grad_phi, force):

    # obtain relevant variables at time n
    dt = cell.simbox.dt
    phi_i = cell.phi
    vx_i = cell.vx
    vy_i = cell.vy

    # compute equation of motion at time n
    dF_dphi = force.total_func_deriv(cell)
    grad_x, grad_y = grad_phi
    v_dot_gradphi = vx_i * grad_x + vy_i * grad_y

    return phi_i - dt * (dF_dphi + v_dot_gradphi), dF_dphi


def _poisson_add_time(rng, rate):
    tau_add = rng.poisson(rate)
    while tau_add == 0:
        tau_add = rng.poisson(rate)
    return tau_add
