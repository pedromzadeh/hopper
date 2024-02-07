from multiprocessing import Process
import sys
from analysis.analysis import *
import numpy as np
import pickle
import os


def process_gid(gid, run_id, min_pts=2):
    data_root = "/mnt/c/Users/pedro/OneDrive - Johns Hopkins/Documents/Research/hopper/_server/sim_data/defaults"

    nbins = 32
    xva_df = get_xva_df(
        apply_time_filter(
            bootstrap(
                read_fulltake(
                    os.path.join(data_root, f"parquets/fulltake_gid{gid}.parquet"),
                    scale_position=True,
                ),
            ),
            dt=3,
            base_rate=3,
        ),
        nbins,
        yfile=os.path.join(data_root, f"configs/grid_id{gid}/simbox.yaml"),
    )

    bounds, F, _ = compute_F_sigma(xva_df, nbins=nbins, min_pts=min_pts)
    xmin, xmax = bounds["x"]
    vmin, vmax = bounds["v"]
    bounds_tup = (xmin, xmax, vmin, vmax, nbins)

    # --- need the entire lattice filled --- #
    dx = (xmax - xmin) / nbins
    dv = (vmax - vmin) / nbins
    X, Y = np.meshgrid(np.arange(nbins), np.arange(nbins))
    X = X * dx + xmin + dx / 2
    Y = Y * dv + vmin + dv / 2
    init_pts = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
    labels, end_pts = get_labels(init_pts, X, Y, F, x_L=135, x_R=165)

    pickle.dump(
        {
            "F": F,
            "bounds": bounds_tup,
            "labels": labels,
            "end_pts": np.array(end_pts),
        },
        open(f"bootstrapped_maps/map_{gid}_{run_id}.pkl", "wb"),
    )


if __name__ == "__main__":
    n_batches = 1
    n_workers = 2
    grid_id = int(sys.argv[1])

    for batch_id in range(n_batches):
        processes = [
            Process(
                target=process_gid,
                args=[grid_id, k],
            )
            for k in range(n_workers)
        ]

        # begin each process
        for p in processes:
            p.start()

        # wait to proceed until all finish
        for p in processes:
            p.join()