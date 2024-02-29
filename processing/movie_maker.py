import numpy as np
import os
from glob import glob
import multiprocessing
import sys


def make(gid, rid):
    root = f"../output/IM/grid_id{gid}/run_{rid}/visuals/"
    files = glob(os.path.join(root, "*.png"))
    ids = [int(f.split("/")[-1].split("_")[1].split(".")[0]) for f in files]

    sort_indx = np.argsort(ids)

    for i, indx in enumerate(sort_indx):
        curr_file = files[indx]
        new_file = f"{root}/img_{i}.png"
        cmd = f"mv {curr_file} {new_file} > dump.txt"
        os.system(cmd)

    os.system(f"mkdir -p {gid}")
    cmd = (
        f"ffmpeg -i {root}/img_%d.png "
        "-b:v 4M -s 600x600 -pix_fmt yuv420p -filter:v 'setpts=PTS' "
        f"{gid}/mov_{rid}.mp4 -y -hide_banner -loglevel fatal"
    )
    os.system(cmd)

    os.system("rm dump.txt")


if __name__ == "__main__":
    
    gid = int(sys.argv[1])
    processes = [
        multiprocessing.Process(target=make, args=[gid, rid])
        for rid in range(48)
    ]

    # begin each process
    for p in processes:
        p.start()

    # wait to proceed until all finish
    for p in processes:
        p.join()
