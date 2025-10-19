import cv2
from pathlib import Path
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger()
config = ConfigLoader()

class FaceDetector:
    def __init__(self):
        models_dir = config.get('paths.models_dir', 'models')
        cascade_filename = config.get('detection.haar_cascade_path', 'haarcascade_frontalface_default.xml')
        cascade_path = Path(cascade_filename)
        if not cascade_path.exists():
            cascade_path = Path(models_dir) / cascade_filename
        
        if not cascade_path.exists():
            logger.error(f"Haar Cascade not found at {cascade_path}. Download to models/ folder.")
            raise FileNotFoundError(f"Cascade missing: {cascade_path}")
        
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            logger.error("Cascade loaded but empty—check file integrity.")
            raise ValueError("Cascade loaded but empty—check file integrity.")

    def detect_faces(self, frame, scaleFactor=1.1, min_neighbors=5, min_size=(30, 30)):
        """Detect faces and return list of (x, y, w, h) for high-quality faces."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=scaleFactor, minNeighbors=min_neighbors,
            minSize=min_size, flags=cv2.CASCADE_SCALE_IMAGE
        )

        good_faces = []
        min_sharpness = config.get('preprocessing.min_sharpness_variance', 100)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            sharpness = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
            if sharpness >= min_sharpness:
                good_faces.append((x, y, w, h))
            else:
                logger.warning(f"Low sharpness face rejected: variance {sharpness:.2f}")
        return good_faces
