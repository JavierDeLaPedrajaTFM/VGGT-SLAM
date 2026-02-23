from vggt_slam.map import Map

class MapManager:
    def __init__(self):
        self.map = Map()

    def integrate(self, frame, pose):
        # Integración mínima usando el mapa del repo original
        self.map.integrate(frame, pose)

    def get_pointcloud(self):
        return self.map.get_pointcloud()
