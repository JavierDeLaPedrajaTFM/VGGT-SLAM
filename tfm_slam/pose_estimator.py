import numpy as np
from vggt.models.vggt import VGGT

class PoseEstimator:
    def __init__(self):
        # Cargar modelo VGGT del repo original
        self.model = VGGT()
        self.model.eval()
        self.prev_frame = None

    def estimate(self, frame):
        # Primer frame → no hay movimiento
        if self.prev_frame is None:
            self.prev_frame = frame
            return np.eye(4)

        # TODO: implementar estimación real usando VGGT
        pose = np.eye(4)

        self.prev_frame = frame
        return pose
