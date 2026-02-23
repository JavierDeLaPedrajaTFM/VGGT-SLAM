import numpy as np
import cv2

class PoseEstimator:
    def __init__(self, focal=800.0, pp=None):
        self.prev_gray = None
        self.K = None
        self.focal = focal
        self.pp = pp  # punto principal (cx, cy), si quieres fijarlo

    def _init_intrinsics(self, frame):
        h, w = frame.shape[:2]
        cx = w / 2.0 if self.pp is None else self.pp[0]
        cy = h / 2.0 if self.pp is None else self.pp[1]
        self.K = np.array([[self.focal, 0, cx],
                           [0, self.focal, cy],
                           [0, 0, 1]], dtype=np.float64)

    def estimate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.K is None:
            self._init_intrinsics(frame)

        if self.prev_gray is None:
            self.prev_gray = gray
            return np.eye(4, dtype=np.float64)

        # Detectar y describir puntos con ORB
        orb = cv2.ORB_create(2000)
        kp1, des1 = orb.detectAndCompute(self.prev_gray, None)
        kp2, des2 = orb.detectAndCompute(gray, None)

        if des1 is None or des2 is None:
            self.prev_gray = gray
            return np.eye(4, dtype=np.float64)

        # Emparejar descriptores
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 8:
            self.prev_gray = gray
            return np.eye(4, dtype=np.float64)

        matches = sorted(matches, key=lambda x: x.distance)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Matriz esencial y pose relativa
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            self.prev_gray = gray
            return np.eye(4, dtype=np.float64)

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)

        # Construir matriz 4x4
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()

        self.prev_gray = gray
        return T
