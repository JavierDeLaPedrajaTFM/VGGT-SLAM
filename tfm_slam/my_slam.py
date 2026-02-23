import numpy as np
from tu_slam.pose_estimator import PoseEstimator
from tu_slam.map_manager import MapManager
from vggt_slam.graph import PoseGraph

class MySLAM:
    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.map_manager = MapManager()
        self.graph = PoseGraph()

    def track(self, frame):
        pose = self.pose_estimator.estimate(frame)
        self.graph.add_pose(pose)
        self.map_manager.integrate(frame, pose)

    def get_trajectory(self):
        return self.graph.get_trajectory()

    def get_pointcloud(self):
        return self.map_manager.get_pointcloud()
