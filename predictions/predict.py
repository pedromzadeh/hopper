import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
import sys

from analysis.analysis import *
from analysis.predict import PredictTrajectory


def worker(process_index, gid, pbar, func, x0, v0, T):
    res = func(x0, v0, T)
    pd.DataFrame(res, columns=["x"]).to_parquet(
        f"pred_trajs/grid_id{gid}_rid{process_index}.parquet"
    )
    pbar.update(1)


if __name__ == "__main__":
    root = "defaults"
    gid = 2

    grid_df = read_fulltake(
        f"../_server/sim_data/{root}/parquets/fulltake_gid{gid}.parquet",
        scale_position=True,
    )

    rids = grid_df.rid.unique()
    train_rids = np.random.choice(rids, size=rids.size // 2, replace=False)
    train_df = grid_df.query("rid in @train_rids").reset_index(drop=True)
    test_df = grid_df.query("rid not in @train_rids").reset_index(drop=True)

    train_df = apply_time_filter(train_df, dt=3, base_rate=3)
    nbins = 32
    train_xva = get_xva_df(
        train_df,
        nbins,
        yfile=f"../_server/sim_data/{root}/configs/grid_id{gid}/simbox.yaml",
    )

    bounds, F, Sigma = compute_F_sigma(train_xva, nbins=nbins, min_pts=10)
    xmin, xmax = bounds["x"]
    vmin, vmax = bounds["v"]

    test_df = apply_time_filter(test_df, dt=3, base_rate=3)
    nbins = 32
    test_xva = get_xva_df(
        test_df,
        nbins,
        yfile=f"../_server/sim_data/{root}/configs/grid_id{gid}/simbox.yaml",
    )
    X = np.linspace(xmin, xmax, nbins + 1)
    V = np.linspace(vmin, vmax, nbins + 1)

    rids = np.random.choice(test_xva.rid.unique(), 10)
    x0v0s = (
        test_xva.reset_index(names="index").query("index==0")[["x", "v"]].values[:100]
    )

    predictor = PredictTrajectory(X, V, F, Sigma)
    T = 240000

    n_workers = multiprocessing.cpu_count() - 3

    print("here")
    with tqdm(total=x0v0s.shape[0] // n_workers) as pbar:
        sub_processes = [
            multiprocessing.Process(
                target=worker,
                args=[k, gid, pbar, predictor.integrate, *x0v0, T],
            )
            for k, x0v0 in enumerate(x0v0s)
        ]

        # begin each process
        for p in sub_processes:
            p.start()

        # wait to proceed until all finish
        for p in sub_processes:
            p.join()

    print("Done!")
