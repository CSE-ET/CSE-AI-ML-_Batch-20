import cv2
import numpy as np
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger()
config = ConfigLoader()

class Preprocessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(
            clipLimit=config.get('preprocessing.clahe_clip_limit', 2.0),
            tileGridSize=tuple(config.get('preprocessing.clahe_tile_grid_size', [8, 8]))
        )

    def preprocess_face(self, face_roi):
        # CLAHE for contrast enhancement
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        
        # Bilateral filtering for noise reduction
        filtered = cv2.bilateralFilter(enhanced, 9, 
                                      config.get('preprocessing.bilateral_sigma_d', 5),
                                      config.get('preprocessing.bilateral_sigma_r', 0.1))
        
        # Simplified geometric normalization (affine alignment to 200x200)
        h, w = filtered.shape
        if max(h, w) > 0:
            normalized = cv2.resize(filtered, (200, 200))
        else:
            normalized = filtered  # Fallback
        
        return normalized