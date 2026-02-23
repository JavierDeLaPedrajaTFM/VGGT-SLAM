from tfm_slam.pose_estimator import PoseEstimator
from tfm_slam.map_manager import MapManager
from vggt_slam.graph import PoseGraph

class MySLAM:
    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.map_manager = MapManager()
        self.graph = PoseGraph()
        self.current_pose = None

    def track(self, frame):
        # Pose relativa entre frames
        T_rel = self.pose_estimator.estimate(frame)

        if self.current_pose is None:
            self.current_pose = T_rel
        else:
            # Componer poses: P_k = P_{k-1} * T_rel
            self.current_pose = self.current_pose @ T_rel

        self.graph.add_pose(self.current_pose)
        self.map_manager.integrate(frame, self.current_pose)

    def get_trajectory(self):
        return self.graph.get_trajectory()

    def get_pointcloud(self):
        return self.map_manager.get_pointcloud()
