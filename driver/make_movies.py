import numpy as np
import os
from glob import glob


rid = [0]
for grid_id in range(8):
    for run_id in rid:
        root = f"../output/IM/grid_id{grid_id}/run_{run_id}/visuals"
        files = glob(os.path.join(root, "*.png"))

        ids = [int(f.split("/")[-1].split("_")[1].split(".")[0]) for f in files]

        sort_indx = np.argsort(ids)

        for i, indx in enumerate(sort_indx):
            curr_file = files[indx]
            new_file = f"{root}/img_{i}.png"
            cmd = f"mv {curr_file} {new_file} > dump.txt"
            os.system(cmd)

        print(f"Making a movie for grid {grid_id} and run {run_id}...")
        cmd = (
            "ffmpeg -i "
            f"{root}/img_%d.png -b:v 4M -s 600x600 -pix_fmt yuv420p -filter:v 'setpts=1.5*PTS' "
            f"mov_{run_id}_{grid_id}.mp4 -y -hide_banner -loglevel fatal"
        )
        os.system(cmd)

os.system("rm dump.txt")
