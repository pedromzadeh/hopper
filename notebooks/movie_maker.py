import numpy as np
import os
from glob import glob


gid = 142
root = f"../output/IM/grid_id{gid}/run_0/visuals/"
files = glob(os.path.join(root, "*.png"))
ids = [int(f.split("/")[-1].split("_")[1].split(".")[0]) for f in files]

sort_indx = np.argsort(ids)

for i, indx in enumerate(sort_indx):
    curr_file = files[indx]
    new_file = f"{root}/img_{i}.png"
    cmd = f"mv {curr_file} {new_file} > dump.txt"
    os.system(cmd)

cmd = (
    f"ffmpeg -i {root}/img_%d.png "
    "-b:v 4M -s 600x600 -pix_fmt yuv420p -filter:v 'setpts=1.5*PTS' "
    f"mov_{gid}.mp4 -y -hide_banner -loglevel fatal"
)
os.system(cmd)

os.system("rm dump.txt")
