from multiprocessing import Pool, cpu_count
from analysis.analysis import *
import numpy as np
import pickle
import os


def process_gid(gid, min_pts=2):
    data_root = "/mnt/c/Users/pedro/OneDrive - Johns Hopkins/Documents/Research/hopper/_server/sim_data/defaults"

    nbins = 32
    xva_df = get_xva_df(
        apply_time_filter(
            read_fulltake(
                os.path.join(data_root, f"parquets/fulltake_gid{gid}.parquet"),
                scale_position=True,
            ),
            dt=3,
            base_rate=3,
        ),
        nbins,
        yfile=os.path.join(data_root, f"configs/grid_id{gid}/simbox.yaml"),
    )

    bounds, F, sigma = compute_F_sigma(xva_df, nbins=nbins, min_pts=min_pts)
    xmin, xmax = bounds["x"]
    vmin, vmax = bounds["v"]
    bounds_tup = (xmin, xmax, vmin, vmax, nbins)
    labels = []
    end_pts = []
    X, Y, init_pts = full_lattice(F, xmin, xmax, vmin, vmax, nbins)
    labels, end_pts = get_labels(init_pts, X, Y, F, x_L=135, x_R=165)

    pickle.dump(
        {
            "F": F,
            "bounds": bounds_tup,
            "labels": labels,
            "end_pts": np.array(end_pts),
        },
        open(f"maps/map_{gid}.pkl", "wb"),
    )


if __name__ == "__main__":
    pool = Pool(processes=cpu_count() - 2)
    args = [
        10,
        26,
        2,
        122,
        145,
        132,
        133,
        136,
        137,
        112,
        114,
        113,
        115,
        139,
        141,
        142,
        144,
        145,
        120,
        146,
        121,
        100,
        4,
        13,
    ]
    # args = [108, 110]  # min_pts = 1; dt = 4, base_rate = 2
    args = [108, 110]  # not anymore
    args = [149, 150]
    args = [151]
    args = [101]
    pool.map(process_gid, args)
