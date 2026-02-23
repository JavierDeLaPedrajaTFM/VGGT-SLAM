import argparse
import numpy as np
import os

from tu_slam.my_slam import MySLAM
from tu_slam.utils import extract_frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    frames = extract_frames(args.input)

    slam = MySLAM()

    for frame in frames:
        slam.track(frame)

    traj = slam.get_trajectory()
    pcd = slam.get_pointcloud()

    np.savetxt(os.path.join(args.output, "trajectory.txt"), traj)
    pcd.export(os.path.join(args.output, "pointcloud.ply"))

if __name__ == "__main__":
    main()
