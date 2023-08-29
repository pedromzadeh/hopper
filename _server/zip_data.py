import pandas as pd
import os
from glob import glob
import multiprocessing


def write_parquet(gid, save_path):
    data_files = glob(f"../output/IM/grid_id{gid}/run_*/result.csv")
    df_gid = []
    for csv in data_files:
        df = pd.read_csv(csv).drop(columns=["Unnamed: 0"])
        df["gid"] = gid
        df["rid"] = int(csv.split("/")[4].split("_")[1])
        df_gid.append(df)

    if len(df_gid) > 0:
        df_gid = pd.concat(df_gid)
        df_gid.to_parquet(f"{save_path}/fulltake_gid{gid}.parquet")


if __name__ == "__main__":
    save_path = "sim_data/rectangular_only"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    n_gid = 32
    processes = [
        multiprocessing.Process(
            target=write_parquet,
            args=[gid, save_path],
        )
        for gid in range(16, n_gid)
    ]

    # begin each process
    for p in processes:
        p.start()

    # wait to proceed until all finish
    for p in processes:
        p.join()
