from multiprocessing import Pool, cpu_count
from analysis.analysis import read_fulltake, get_itinerary
import numpy as np
import os


def get_hopping_times(parquet_file):
    save_path = os.path.join(
        os.path.dirname(os.path.dirname(parquet_file)), "hopping times"
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    grid_df = read_fulltake(parquet_file, scale_position=True)

    hop_times = []
    for _, df in grid_df.groupby("rid"):
        arrive_depart_indx = get_itinerary(df)
        if arrive_depart_indx is None:
            continue
        for t in [np.ptp(df.iloc[s:e]["time[hr]"]) for s, e, in arrive_depart_indx]:
            hop_times.append(t)

    np.save(
        os.path.join(
            save_path, f"{parquet_file.split('.parquet')[0].split('/')[-1]}.npy"
        ),
        np.array(hop_times),
    )
    return None


if __name__ == "__main__":
    from glob import glob

    pool = Pool(processes=cpu_count() - 1)
    root = "defaults"
    args = glob(f"../_server/sim_data/{root}/parquets/*")
    pool.map(get_hopping_times, args)
