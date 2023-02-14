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


def map_a(binned_df, nbins, x1bin_lbl, x2bin_lbl, min_pts):

    acc_map = np.empty(shape=(nbins, nbins))
    acc_map[:] = np.nan

    # x1 is along the x-axis
    # x2 is along the y-axis
    for (j, i), df in binned_df.groupby([x1bin_lbl, x2bin_lbl]):
        if len(df.a) < min_pts:
            continue
        acc_map[i, j] = df.a.mean()

    return acc_map


def get_bin_indices(df, nbins, x1=None, x2=None):

    assert x1 is not None
    assert x2 is not None

    x1_lbl, x1_min, dx1 = x1
    x2_lbl, x2_min, dx2 = x2

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

    axs[2, 2].remove()
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


def map_imshow(acc_map, ax, bounds, nbins, cbar=False):
    xmin, xmax, vmin, vmax = bounds

    im = ax.imshow(acc_map, origin="lower", interpolation="none", cmap="rainbow")
    ax.set_xticks(
        [0, nbins // 2, nbins],
        [round(xmin, 2), round((xmin + xmax) / 2, 2), round(xmax, 2)],
    )
    ax.set_yticks(
        [0, nbins // 2, nbins],
        [round(vmin, 2), round((vmin + vmax) / 2, 0), round(vmax, 2)],
    )
    if cbar:
        cbar = ax.gcf().colorbar(im, ax=ax)
        cbar.set_label(r"$F$ ($\mu$m/hr$^2$)")
        ax.set_xlabel(r"$x$ ($\mu$m)")
        ax.set_ylabel(r"$v$ ($\mu$m/hr)")
