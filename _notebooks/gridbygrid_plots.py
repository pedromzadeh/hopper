from multiprocessing import Pool, cpu_count
from analysis.analysis import *
import matplotlib.pyplot as plt
import os


def plot_nonlinearity_F(time_filter):
    root = "corners_only_fixed_augmented"
    save_path = f"../_server/sim_data/{root}/temporal_filtering/"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    print(f"Dealing with time filter {time_filter}")
    nbins = 32
    delta = 2
    min_pts = 10

    nrows, ncols = 4, 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 20), dpi=300)

    for gid in range(16):
        grid_df = read_fulltake(
            f"../_server/sim_data/{root}/parquets/fulltake_gid{gid}.parquet",
            scale_position=True,
        )

        if time_filter is None:
            grid_df = apply_time_filter(grid_df, dt=grid_df.iloc[0].add_rate)
        else:
            grid_df = apply_time_filter(grid_df, dt=time_filter)

        xva_df = get_xva_df(grid_df, nbins=nbins)
        bounds, F, sigma = compute_F_sigma(xva_df, nbins=nbins, min_pts=min_pts)
        xmin, xmax = bounds["x"]
        vmin, vmax = bounds["v"]

        X, Y, init_pts = full_lattice(F, xmin, xmax, vmin, vmax, nbins)

        F_v_x0 = F[:, nbins // 2 - delta : nbins // 2 + (delta + 1)]
        mask = np.prod(~np.isnan(F_v_x0), axis=1) == 1
        F_v_x0 = F_v_x0.mean(axis=1)

        i, j = gid // ncols, gid % ncols
        axs[i, j].set_title(make_title(xva_df), fontsize=16)
        axs[i, j].plot(Y[mask, 0], F_v_x0[mask], "-o", c="black")
        axs[i, j].set_xlabel(r"$v$ ($\mu$m/hr)")
        axs[i, j].set_ylabel(r"$F(x\rightarrow 0)$ ($\mu$m/hr$^2$)")

    plt.subplots_adjust(wspace=0.75, hspace=1.5)
    plt.savefig(save_path + f"time_filter_{time_filter}.png")
    plt.close()
    return None


def plot_separatrices(time_filter):
    root = "corners_only_fixed_augmented"
    save_path = f"../_server/sim_data/{root}/temporal_filtering/"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    print(f"Dealing with time filter {time_filter}")
    nbins = 32
    min_pts = 10

    nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 7.5), dpi=300)
    axs[1, 2].remove()

    for k, gid in enumerate([1, 3, 5, 7, 15]):
        grid_df = read_fulltake(
            f"../_server/sim_data/{root}/parquets/fulltake_gid{gid}.parquet",
            scale_position=True,
        )

        if time_filter is None:
            grid_df = apply_time_filter(grid_df, dt=grid_df.iloc[0].add_rate)
        else:
            grid_df = apply_time_filter(grid_df, dt=time_filter)

        xva_df = get_xva_df(grid_df, nbins=nbins)
        bounds, F, sigma = compute_F_sigma(xva_df, nbins=nbins, min_pts=min_pts)
        xmin, xmax = bounds["x"]
        vmin, vmax = bounds["v"]
        bounds_tup = (xmin, xmax, vmin, vmax, nbins)

        X, Y, init_pts = full_lattice(F, xmin, xmax, vmin, vmax, nbins)
        labels = get_labels(init_pts, X, Y, F)
        img = lattice_to_image(init_pts, labels, bounds_tup)
        X = get_separatrices(
            img, levels=[0.5, 1], origin="lower", extent=[xmin, xmax, vmin, vmax]
        )

        i, j = k // ncols, k % ncols
        axs[i, j].set_title(make_title(xva_df), fontsize=16)
        axs[i, j].vlines(
            x=133,
            ymin=vmin,
            ymax=vmax,
            linewidth=1,
            colors=["black"],
            linestyles=["dashed"],
        )
        axs[i, j].vlines(
            x=167,
            ymin=vmin,
            ymax=vmax,
            linewidth=1,
            colors=["black"],
            linestyles=["dashed"],
        )

        axs[i, j].scatter(init_pts[:, 0], init_pts[:, 1], color=labels, s=5)
        axs[i, j].plot(X[:, 0], X[:, 1], lw=2, color="black")
        axs[i, j].set_xlabel(r"$x$ ($\mu$m)")
        axs[i, j].set_ylabel(r"$v$ ($\mu$m/hr)")
        xlim = axs[i, j].get_xlim()
        ylim = axs[i, j].get_ylim()

    plt.subplots_adjust(wspace=0.75, hspace=1.5)
    plt.savefig(save_path + f"separatrices_dt_{time_filter}.png")
    plt.close()
    return None


def plot_Fvx0_bootstraps(gid):
    from time import time

    root = "corners_only_fixed_augmented"
    save_path = f"../_server/sim_data/{root}/bootstraps/"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    print(f"Dealing with grid ID {gid}")
    nbins = 32
    min_pts = 10
    # collect bootstrap samples
    V_samples, Fvx0_samples = [], []

    grid_df = read_fulltake(
        f"../_server/sim_data/{root}/parquets/fulltake_gid{gid}.parquet",
        scale_position=True,
    )
    grid_df = apply_time_filter(grid_df, dt=4)
    xva_df = get_xva_df(grid_df, nbins=nbins)
    bounds, F, sigma = compute_F_sigma(xva_df, nbins=nbins, min_pts=min_pts)
    xmin, xmax = bounds["x"]
    vmin, vmax = bounds["v"]
    X, Y, init_pts = full_lattice(F, xmin, xmax, vmin, vmax, nbins)
    v, f = evaluate_F_v_x0(Y, F, nbins)
    V_samples.append(v)
    Fvx0_samples.append(f)

    # bootstrap
    M = 50
    for k in range(M):
        sample_grid_df = bootstrap(
            grid_df, n_samples=grid_df.rid.unique().size, seed=int(time()) + k
        )
        sample_xva_df = get_xva_df(sample_grid_df, nbins=nbins)
        bounds, F, sigma = compute_F_sigma(sample_xva_df, nbins=nbins, min_pts=min_pts)
        xmin, xmax = bounds["x"]
        vmin, vmax = bounds["v"]
        X, Y, init_pts = full_lattice(F, xmin, xmax, vmin, vmax, nbins)
        v, f = evaluate_F_v_x0(Y, F, nbins)
        V_samples.append(v)
        Fvx0_samples.append(f)

    lim = np.min([np.fabs([v.min(), v.max()]).min() for v in V_samples])
    v_interp = np.linspace(-lim, lim, 30)
    fs_interp = np.array(
        [np.interp(v_interp, v, f) for v, f in zip(V_samples, Fvx0_samples)]
    )
    f_avg = np.mean(fs_interp, axis=0)
    # ci = 1.96 * np.std(fs_interp, axis=0) / fs_interp.shape[0]
    ci = np.std(fs_interp, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
    ax.plot(v_interp, f_avg, "-o", c="black", lw=1)
    ax.plot(-v_interp, -f_avg, c="cornflowerblue", lw=1)
    ax.fill_between(v_interp, f_avg - ci, f_avg + ci, color="salmon", alpha=0.9)
    ax.set_xlabel(r"$v$ ($\mu$m/hr)")
    ax.set_ylabel(r"$F(x\rightarrow 0)$ ($\mu$m/hr$^2$)")
    plt.savefig(save_path + f"F_v_x0_gid_{gid}.png")
    plt.close()
    return None


if __name__ == "__main__":
    pool = Pool(processes=4)
    args = [gid for gid in np.arange(16)]
    args.pop(0)
    args.pop(1)
    pool.map(plot_Fvx0_bootstraps, args)
