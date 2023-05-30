import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("../configs/stylesheet.mplstyle")


def calc_v_a_from_position(x, dt):
    """
    Calculates speed and acceleration from 1D position time series.

    Parameters
    ----------
    x : arraylike
        Time series of x position of center of mass, in microns.

    dt : float
        Time step between the data points, in minutes.

    Returns
    -------
    df : pd.DataFrame
        Columns are "x", "v", "a". The positions are the ones
        associated with speed and acceleration values; i.e. x[:-2].

    Note
    ----
    We're taking numerical derivatives to find these two quantities.
    Speed calculation knocks out x[-1], and acceleration calculation
    knocks out v[-1] -- thereby knocking out x[-2]. Also the convention
    is v(n) = [x(n+1) - x(n)] / dt && a(n) = [v(n+1) - v(n)] / dt.
    """

    def _calc_v():
        return np.diff(x) / np.diff(dt)

    def _calc_a():
        return np.diff(v) / np.diff(dt[:-1])

    v = _calc_v()
    a = _calc_a()
    res = np.array([x[:-2], v[:-1], a])
    return pd.DataFrame(res.T, columns=["x", "v", "a"])


def calc_F_sigma(binned_df, dt, nbins, min_pts):

    F = np.empty(shape=(nbins, nbins))
    F[:] = np.nan
    F_std = np.empty(shape=(nbins, nbins))
    F_std[:] = np.nan

    sigma = np.empty(shape=(nbins, nbins))
    sigma[:] = np.nan

    # F(x, v)
    for (j, i), df in binned_df.groupby(["x_bin", "v_bin"]):
        if len(df.a) < min_pts:
            continue
        F[i, j] = df.a.mean()
        F_std[i, j] = df.a.std() / df.a.size
        sigma[i, j] = np.sqrt(dt * np.mean((df.a.values - F[i, j]) ** 2))

    return F, F_std, sigma


def get_bin_indices(df, nbins):

    # define binned space
    bounds = df.agg(["min", "max"])
    xmin, xmax = bounds["x"]
    vmin, vmax = bounds["v"]
    dx = (xmax - xmin) / nbins
    dv = (vmax - vmin) / nbins

    x1_lbl, x1_min, dx1 = ("x", xmin, dx)
    x2_lbl, x2_min, dx2 = ("v", vmin, dv)

    assert set([x1_lbl, x2_lbl]).issubset(df.columns)

    df[f"{x1_lbl}_bin"] = np.floor((df[x1_lbl].values - x1_min) / dx1).astype("int")
    df[f"{x2_lbl}_bin"] = np.floor((df[x2_lbl].values - x2_min) / dx2).astype("int")

    # max values should be dropped by 1 in index value
    df[f"{x1_lbl}_bin"] = df[f"{x1_lbl}_bin"].apply(
        lambda x: x - 1 if x == nbins else x
    )
    df[f"{x2_lbl}_bin"] = df[f"{x2_lbl}_bin"].apply(
        lambda x: x - 1 if x == nbins else x
    )


def get_hopping_times(df, xL, xR, buffer):
    """
    Return hopping times present in given input in hours.

    Parameters
    ----------
    df : pd.DataFrame
        Time series of positions, in microns.

    xL, xR : float
        Centroids of the two basins, in microns.

    buffer : float
        Tolerance away from basin centroid to count inside the basin, in microns.

    mu_factor : float
        Specifies conversion factor between simulation
        units and microns.

    Returns
    -------
    ndarray
        The hopping times in hours.
    """
    # returns hopping times for this run in hours
    def _state(x):
        # x /= mu_factor
        if np.fabs(x - xL) < buffer:
            return 0
        elif np.fabs(x - xR) < buffer:
            return 1
        else:
            return -1

    df["state"] = df.x.apply(_state)
    df = df.query("state != -1")
    swaps = np.diff(df.state)
    swap_inxs = np.where(swaps != 0)[0] + 1
    times = df["time[hr]"].iloc[swap_inxs].to_numpy()
    hopping_times = np.diff(times, prepend=0)
    return hopping_times


def position_dist(data, mp_type, mu_factor, cmap=None):
    from substrate.substrates import Substrate

    sub_generator = Substrate(N_mesh=200, L_box=50)
    if mp_type == "two_state":
        chi = sub_generator.two_state_sub()
    elif mp_type == "rectangular":
        chi = sub_generator.rectangular()
    else:
        raise KeyError(f"{mp_type} is not understood")

    fig, axs = plt.subplots(3, 3, figsize=(5, 5), dpi=200)

    x, y = np.meshgrid([0, 1, 2], [0, 1, 2])

    for i, j, df in zip(x.flatten(), y.flatten(), data):
        beta = df.beta.iloc[0]
        gamma = df.gamma.iloc[0]
        D = df.D.iloc[0]
        gid = df.gid.iloc[0]
        axs[i, j].set_title(
            rf"$\beta$ = {beta}, $\gamma$ = {gamma}, $D$ = {D}" "\n" f"[{gid}]"
        )
        axs[i, j].contour(
            chi,
            levels=[0.5],
            colors=["black"],
            linewidths=[3],
            extent=[0, 50 * mu_factor, 0, 50 * mu_factor],
        )
        axs[i, j].scatter(df.x, df.y, color="orange", s=3)
        axs[i, j].axis("off")
        axs[i, j].set_xlim([15 * mu_factor, 35 * mu_factor])
        axs[i, j].set_ylim([15 * mu_factor, 35 * mu_factor])

    # remove axes that are empty
    for i, j in zip(x.flatten(), y.flatten()):
        if not axs[i, j].collections:
            axs[i, j].remove()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()


def streamplot(
    acc_map,
    ax,
    bounds,
    nbins,
    init_cond=False,
    init_type="grid",
    n_skip=None,
    one_side=None,
):
    xmin, xmax, vmin, vmax = bounds
    midpoint = (xmax + xmin) / 2
    buffer = 0.5
    X, Y = np.meshgrid(
        np.linspace(xmin + buffer, xmax - buffer, nbins),
        np.linspace(vmin + buffer, vmax - buffer, nbins),
    )

    # baseline streams
    ax.streamplot(
        X,
        Y,
        Y,
        acc_map,
        linewidth=0.5,
        integration_direction="forward",
        color="gainsboro",
        density=2,
    )

    # if initial conditions are given
    if init_cond:
        fig_temp, ax_temp = plt.subplots(1, 1)

        if init_type == "grid":
            non_nans = np.argwhere(~np.isnan(acc_map))
            x, y = X[0][non_nans[:, 1]], Y[:, 0][non_nans[:, 0]]
        elif init_type == "linear":
            x, y = np.linspace(xmin, xmax, 500), np.linspace(vmin, vmax, 500)
        else:
            raise KeyError(f"{init_type} not understood.")

        if n_skip is not None:
            x, y = x[::n_skip], y[::n_skip]

        # draw flow lines for all points falling on non-nan acc values
        for xx, yy in zip(x, y):
            try:
                stream = ax_temp.streamplot(
                    X,
                    Y,
                    Y,
                    acc_map,
                    start_points=[[xx, yy]],
                    integration_direction="forward",
                    broken_streamlines=True,
                    density=10,
                )
                traj = np.array(stream.lines.get_segments())
            except (ValueError):
                print(f"Points {[xx, yy]} outside bounds.")
                continue

            if len(traj) == 0:
                continue

            flag = traj[-1][-1][0] < midpoint
            if flag:
                color = "red"
            else:
                color = "blue"

            if one_side is not None:
                if color == one_side:
                    ax.streamplot(
                        X,
                        Y,
                        Y,
                        acc_map,
                        linewidth=0.5,
                        start_points=[[xx, yy]],
                        integration_direction="forward",
                        color=color,
                        broken_streamlines=True,
                        density=10,
                    )
                else:
                    continue
            else:
                ax.streamplot(
                    X,
                    Y,
                    Y,
                    acc_map,
                    linewidth=0.5,
                    start_points=[[xx, yy]],
                    integration_direction="forward",
                    color=color,
                    broken_streamlines=True,
                    density=10,
                )
        plt.close(fig_temp)


def imshow_F_sigma(maps, bounds, title, interp="none", err_cutoff=None, save_path=None):

    fig1, axs = plt.subplots(1, 3, figsize=(15, 3.5), dpi=300)
    xmin, xmax, vmin, vmax, nbins = bounds
    axs[0].set_title(title["title"], fontsize=title["size"], y=1.1)
    axs[0].set_xlabel(r"$x$ ($\mu$m)")
    axs[0].set_ylabel(r"$v$ ($\mu$m/hr)")

    F, sigma = maps

    for field, cbar_title, cmap, ax in zip(
        maps,
        [r"$F$ ($\mu$m/hr$^2$)", r"$\sigma$ ($\mu$m hr$^{-3/2}$)"],
        ["jet", "viridis"],
        axs,
    ):
        im = ax.imshow(
            field,
            origin="lower",
            interpolation=interp,
            cmap=cmap,
            extent=[xmin, xmax, vmin, vmax],
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(cbar_title)
        ax.set_aspect("auto")
        # ax.set_xlabel(r"$x$ ($\mu$m)")
        # ax.set_ylabel(r"$v$ ($\mu$m/hr)")

    # plot streamplots for F(x, v)
    if err_cutoff is not None:
        _mask = (sigma / F) * 100 > err_cutoff
        F[_mask] = np.nan
    _plot_trajs(F, bounds, axs[2])

    fig1.subplots_adjust(wspace=0.75)
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def _plot_trajs(F, bounds, ax):
    xmin, xmax, vmin, vmax, nbins = bounds

    # make stremplot for F(x, v)
    buffer = 0.5
    X, Y = np.meshgrid(
        np.linspace(xmin + buffer, xmax - buffer, nbins),
        np.linspace(vmin + buffer, vmax - buffer, nbins),
    )

    x = np.linspace(xmin, xmax, 100)
    y = (vmax - vmin) / (xmax - xmin) * (x - xmin) + vmin

    # if initial condition exists in F, make trajectory
    for xx, yy in zip(x, y):
        try:
            ax.streamplot(
                X,
                Y,
                Y,
                F,
                start_points=[[xx, yy]],
                integration_direction="forward",
                broken_streamlines=True,
                density=10,
                color="black",
            )
        except:
            continue

    ax.quiver(
        X, Y, Y, F, F, angles="xy", scale_units="xy", scale=80, width=0.01, cmap="jet"
    )

    ax.set_aspect("auto")
    # ax.set_xlabel(r"$x$ ($\mu$m)")
    # ax.set_ylabel(r"$v$ ($\mu$m/hr)")
