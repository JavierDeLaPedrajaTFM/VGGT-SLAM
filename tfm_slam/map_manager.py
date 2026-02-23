import numpy as np
import open3d as o3d

class MapManager:
    def __init__(self):
        self.points = []

    def integrate(self, frame, pose):
        # De momento no generamos puntos reales (eso vendrá después)
        # Solo guardamos la posición de la cámara como "punto"
        cam_pos = pose[:3, 3]
        self.points.append(cam_pos.copy())

    def get_pointcloud(self):
        if len(self.points) == 0:
            return o3d.geometry.PointCloud()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(self.points))
        return pcd
