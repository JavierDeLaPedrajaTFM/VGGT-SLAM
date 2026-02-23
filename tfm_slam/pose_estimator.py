import numpy as np
from vggt.models.vggt import VGGT

class PoseEstimator:
    def __init__(self):
        self.model = VGGT()
        self.model.eval()
        self.prev_frame = None

    def estimate(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame
            return np.eye(4)

        # Aquí usarías el modelo VGGT para obtener correspondencias
        # y estimar la pose. De momento devolvemos identidad.
        pose = np.eye(4)

        self.prev_frame = frame
        return pose
