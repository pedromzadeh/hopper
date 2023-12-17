import pandas as pd
import numpy as np
import yaml


L_box = 50
mu_factor = 6  # conversion to microns
min_factor = 8  # conversion to minutes


def read_fulltake(filename, scale_position=False):
    """
    Read sim data file.

    Parameters
    ----------
    filename : str
        Full path to data.

    scale_position : bool, optional
        If True, will convert position to microns, by default False

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    NotImplementedError
        If file extension not .parquet or .pickle.
    """
    ext = filename.split(".")[-1]
    if ext == "parquet":
        df = pd.read_parquet(filename)
    elif ext == "pickle":
        df = pd.read_pickle(filename)
    else:
        raise NotImplementedError(f"Reading file extension {ext} not implemented.")

    if scale_position:
        df.x *= mu_factor
        df.y *= mu_factor
    return df


def bootstrap(df, n_samples, seed=None):
    """
    Generate samples from source with replacement.

    Parameters
    ----------
    df : pd.DataFrame
        The original full sim record.

    n_samples : int
        Number of runs to sample.

    seed : int, optional
        Set for reproducibility.

    Returns
    -------
    pd.DataFrame
        Style is identical to df.
    """
    rng = np.random.default_rng(seed=seed)
    df_list = [elem.reset_index(drop=True) for _, elem in df.groupby("rid")]
    return pd.concat(
        [
            df_list[k]
            for k in rng.choice(np.arange(len(df_list)), replace=True, size=n_samples)
        ]
    )


def apply_time_filter(df, dt, base_rate):
    """
    Filter records based on a time window.

    Parameters
    ----------
    df : pd.DataFrame
        Original record.

    dt : float
        Time window in minutes for filtering sim data.

    base_rate : float
        Rate in minutes at which sim data has been collected.

    Returns
    -------
    pd.DataFrame
        Time filtered records.

    Raises
    ------
    RuntimeError
        Ensures time window is compatible with base rate.
    """
    if dt % base_rate != 0:
        raise RuntimeError(
            "Time window does not neatly fit given base rate of data collection."
        )

    df["ts"] = np.round(df["time[hr]"].values * 60) // dt
    return df.drop_duplicates(subset=["ts", "rid"], keep="first").reset_index(drop=True)


def linear_lattice(xmin, xmax, vmin, vmax, n_pts, s=1, basin_only=False):
    """
    Generates [x, y] points linearly spaced over the xv- phase-space.

    Parameters
    ----------
    xmin, xmax, vmin, vmax : float
        Bounds of the xv- phase-space.

    n_pts : int
        Number of points to generate.

    s : int, optional
        Slope of the line, by default 1

    basin_only : bool, optional
        If True, points are inside the basins only, by default False

    Returns
    -------
    np.ndarray of shape (n_pts, 2)
    """

    def _around_basin():
        d = 10
        x1 = np.linspace(xmin - d, xmin + d, n_pts)
        x2 = np.linspace(xmax - d, xmax + d, n_pts)
        x = np.append(x1, x2)
        y = s * (vmax - vmin) / (xmax - xmin) * (x - xmin) + vmin
        return np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])

    if basin_only:
        return _around_basin()

    x = np.linspace(xmin, xmax, n_pts)
    y = s * (vmax - vmin) / (xmax - xmin) * (x - xmin) + vmin
    return np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])


def full_lattice(F, xmin, xmax, vmin, vmax, nbins):
    """
    Generate [x, y] lattice sites everwhere F is defined.

    Parameters
    ----------
    F : np.ndarray of shape (nbins, nbins)
        The footprint phase-space.

    xmin, xmax, vmin, vmax : float
        Bounds of the xv- phase-space.

    nbins : int
        Bin size.

    Returns
    -------
    np.ndarray of shape (n_pts, 2)
    """
    dx = (xmax - xmin) / nbins
    dv = (vmax - vmin) / nbins
    X, Y = np.meshgrid(np.arange(nbins), np.arange(nbins))
    X = X * dx + xmin + dx / 2
    Y = Y * dv + vmin + dv / 2

    non_nans = np.argwhere(~np.isnan(F))
    x, y = X[0][non_nans[:, 1]], Y[:, 0][non_nans[:, 0]]
    return X, Y, np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])


def get_xva_df(fulltake_df, nbins, yfile=None):
    def _get_mp_type(yfile):
        with open(yfile) as f:
            return yaml.safe_load(f)["substrate"]["kind"]

    grid_x_v_a = []
    for rid, df_rid in fulltake_df.groupby("rid"):
        x_v_a = calc_v_a_from_position(df_rid.x, df_rid["time[hr]"])
        x_v_a["time[hr]"] = df_rid.reset_index()["time[hr]"][:-2]
        x_v_a["rid"] = int(rid)
        grid_x_v_a.append(x_v_a)

    grid_x_v_a = pd.concat(grid_x_v_a)
    features = ["gamma", "R_eq", "sigma_mvg", "tau_mvg", "gid"]
    d = dict(fulltake_df.iloc[0])
    for key in features:
        grid_x_v_a[key] = d[key]
    grid_x_v_a["substrate"] = _get_mp_type(yfile)

    bounds = grid_x_v_a.agg(["min", "max"])
    xmin, xmax = bounds["x"]
    vmin, vmax = bounds["v"]

    get_bin_indices(grid_x_v_a, nbins, xmin, xmax, vmin, vmax)

    return grid_x_v_a


def compute_F_sigma(grid_xva_df, nbins, min_pts):
    F = np.empty(shape=(nbins, nbins))
    F[:] = np.nan

    sigma = np.empty(shape=(nbins, nbins))
    sigma[:] = np.nan
    dt = np.mean(
        [np.mean(np.diff(x["time[hr]"])) for _, x in grid_xva_df.groupby("rid")]
    )

    # F(x, v)
    for (j, i), df in grid_xva_df.groupby(["x_bin", "v_bin"]):
        if len(df.a) < min_pts:
            continue
        F[i, j] = df.a.mean()
        sigma[i, j] = np.sqrt(dt * np.mean((df.a.values - F[i, j]) ** 2))

    bounds = grid_xva_df.agg(["min", "max"])
    return bounds, F, sigma


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


def get_bin_indices(df, nbins, xmin, xmax, vmin, vmax):
    # define binned space
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


def get_boundary_points(pts, labels):
    df = pd.DataFrame(
        np.hstack([pts, np.array(labels).reshape(-1, 1)]), columns=["x", "y", "label"]
    )
    df = df[df.label != "gainsboro"]
    encoding = {"wheat": 2, "salmon": 1, "cornflowerblue": 3}
    df["label"] = df.apply(lambda x: encoding[x.label], axis=1)
    df.x = df.x.astype(float)
    df.y = df.y.astype(float)
    df.label = df.label.astype(int)
    df = df.sort_values(["x", "y"]).reset_index(drop=True)

    Z = []
    for x, df_x in df.groupby("x"):
        k = np.where(np.diff(df_x.label) != 0)[0]
        if len(k):
            Z.append(df.iloc[k + df_x.index[0]][["x", "y"]])

    df = df.sort_values(["y", "x"]).reset_index(drop=True)
    for y, df_y in df.groupby("y"):
        k = np.where(np.diff(df_y.label) != 0)[0]
        if len(k):
            Z.append(df.iloc[k + df_y.index[0]][["x", "y"]])

    Z = pd.concat(Z).drop_duplicates()
    return Z.values


def TSP(pts, dx, dy):
    def _distance(x, ys, dx, dy):
        return np.sqrt(
            [(x[0] - y[0]) ** 2 + ((dx / dy) * (x[1] - y[1])) ** 2 for y in ys]
        )

    Z = sorted(pts, key=lambda x: [x[0], -x[1]])
    m = 0
    sorted_pts = []

    while len(Z):
        curr = Z[m]
        sorted_pts.append(Z[m])
        Z.pop(m)
        dist = _distance(curr, Z, dx, dy)
        try:
            m = np.argsort(dist)[0]
        except IndexError:
            m = np.argsort(dist)

    return np.array(sorted_pts)


def break_upper_lower(pts):
    upper_pts = (
        pd.DataFrame(pts, columns=["x", "y"])
        .sort_values(["x", "y"])
        .drop_duplicates("x", keep="last")
        .values
    )
    upper_pts = np.insert(upper_pts, upper_pts.shape[0], [[np.nan, np.nan]], axis=0)

    lower_pts = (
        pd.DataFrame(pts, columns=["x", "y"])
        .sort_values(["x", "y"])
        .drop_duplicates("x", keep="first")
        .values
    )

    return np.insert(upper_pts, upper_pts.shape[0], lower_pts, axis=0)


def make_title(df):
    tbl = {
        "gamma": r"$\gamma$",
        "R_eq": r"$R_{eq}$",
        "sigma_mvg": r"$\sigma_{MVG}$",
        "tau_mvg": r"$\tau_{MVG}$",
        "gid": "ID",
        "substrate": "Substrate",
    }
    d = dict(df.iloc[0])
    title = ""
    for key, _ in tbl.items():
        try:
            val = np.round(d[key] * 8 * 60, 2) if key == "tau_mvg" else d[key]
            title += tbl[key] + " = " + f"{val}" + "\n"
        except KeyError:
            continue
    return title


def F_streamplot(
    F,
    bounds,
    stream_init_pts,
    imshow_kwargs,
    streamplot_kwargs,
    do_try=False,
    dpi=300,
    ax=None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), dpi=dpi)

    xmin, xmax, vmin, vmax, _ = bounds
    ax.set_xlabel(r"$x$ ($\mu$m)")
    ax.set_ylabel(r"$v$ ($\mu$m/hr)")

    im = ax.imshow(F, extent=[xmin, xmax, vmin, vmax], **imshow_kwargs)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$F$ ($\mu$m/hr$^2$)")
    ax.set_aspect("auto")

    _plot_streams(F, bounds, stream_init_pts, ax, do_try, streamplot_kwargs)


def evaluate_F_v_x0(V, F, nbins, delta=1):
    F_v_x0 = F[:, nbins // 2 - delta : nbins // 2 + delta]
    F_v_x0 = F_v_x0.mean(axis=1)
    mask = ~np.isnan(F_v_x0)
    return V[mask, 0], F_v_x0[mask]


def get_labels(pts, X, Y, F):
    import matplotlib.pyplot as plt

    colors = []
    end_pts = []

    for xx, yy in pts:
        fig, ax = plt.subplots(1, 1)
        try:
            stream = ax.streamplot(
                X,
                Y,
                Y,
                F,
                linewidth=0.5,
                start_points=[[xx, yy]],
                integration_direction="forward",
                color="black",
                broken_streamlines=False,
                density=1,
            )
            streamlines = np.array(stream.lines.get_segments())
            plt.close(fig)

            if len(streamlines) == 0:
                colors.append("gainsboro")
                continue

            end_pt = streamlines[-1][-1]
            end_pts.append(end_pt)
            # if end_pt[0] < 133:
            if end_pt[0] < 145:
                colors.append("salmon")
            # elif end_pt[0] > 167:
            elif end_pt[0] > 155:
                colors.append("cornflowerblue")
            else:
                colors.append("wheat")

        except ValueError:
            colors.append("gainsboro")
            plt.close(fig)
            continue

    return colors, end_pts


def lattice_to_image(xv_pts, labels, bounds):
    xmin, xmax, vmin, vmax, nbins = bounds

    legend_dict = {"salmon": 2, "cornflowerblue": 0, "wheat": 1, "gainsboro": np.nan}
    img = np.ones(shape=(nbins, nbins)) * np.nan
    xv = pd.DataFrame(xv_pts, columns=["x", "v"])
    get_bin_indices(xv, nbins, xmin, xmax, vmin, vmax)
    indx = xv[["v_bin", "x_bin"]].values
    for ij, val in zip(indx, [legend_dict[lbl] for lbl in labels]):
        img[ij[0], ij[1]] = val

    return img


def get_separatrices(image, **contour_kwargs):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    paths = ax.contour(image, **contour_kwargs)
    plt.close(fig)

    paths = [paths.collections[i].get_paths() for i in range(len(paths.collections))]

    X = []
    for path in paths:
        for p in path:
            X.append(p.vertices)
            X.append([np.nan, np.nan])
    X = np.vstack(X)
    return X


def assign_basin_id(x, xL, xR):
    if x < xL:
        return 0
    elif x > xR:
        return 1
    else:
        return -1


def group_consecutive_values(arr):
    res_indx = []
    res = []

    curr_indx = 0
    while curr_indx < len(arr) - 1:
        if arr[curr_indx] == arr[curr_indx + 1]:
            curr_indx += 1
            continue

        res_indx.append([curr_indx, curr_indx + 1])
        res.append([arr[curr_indx], arr[curr_indx + 1]])
        curr_indx += 1

    return np.array(res_indx), np.array(res)


def get_itinerary(df, xL=133, xR=167, ax=None, plot_segments=False):
    states = df.x.apply(assign_basin_id, xL=xL, xR=xR).values
    change_indx, change_vals = group_consecutive_values(states)
    if change_vals.shape[0] == 0:
        return None

    change_time_table = np.insert(change_vals, 2, change_indx[:, 0], axis=1)
    arrive_depart_table = []
    for i in np.arange(0, change_time_table.shape[0], 2):
        chunk = change_time_table[i : i + 2]
        if np.unique(chunk[:, :2]).shape[0] == 3:
            arrive_depart_table.append([chunk[0, -1], chunk[1, -1]])

    q = np.unique(np.array(arrive_depart_table).reshape(-1, 2), axis=0)
    q = np.insert(q, 0, 0)
    q = np.insert(q, q.shape[0], df.index[-1]).reshape(-1, 2).astype(int)

    if plot_segments:
        assert ax is not None
        [ax.plot(df.iloc[s:e].x, df.iloc[s:e].y, alpha=0.5) for s, e in q]
        ax.set_ylim((140, 160))
        ax.vlines([xL, xR], *ax.get_ylim(), linestyles=["dashed"], colors=["black"])

    return q


def compute_cdf(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return x.value_counts().sort_index().cumsum() / x.size


def _plot_streams(F, bounds, init_pts, ax, do_try, streamplot_kwargs):
    xmin, xmax, vmin, vmax, nbins = bounds

    # make stremplot for F(x, v)
    dx = (xmax - xmin) / nbins
    dv = (vmax - vmin) / nbins
    X, Y = np.meshgrid(np.arange(nbins), np.arange(nbins))
    X = X * dx + xmin + dx / 2
    Y = Y * dv + vmin + dv / 2

    if not do_try:
        ax.streamplot(X, Y, Y, F, start_points=init_pts, **streamplot_kwargs)
    else:
        # if initial condition exists in F, make trajectory
        for xx, yy in init_pts:
            try:
                ax.streamplot(X, Y, Y, F, start_points=[[xx, yy]], **streamplot_kwargs)
            except ValueError:
                continue

    ax.quiver(
        X, Y, Y, F, F, angles="xy", scale_units="xy", scale=80, width=0.01, cmap="jet"
    )

    ax.set_aspect("auto")
