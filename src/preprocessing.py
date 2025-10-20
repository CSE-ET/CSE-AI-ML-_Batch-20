import cv2
import numpy as np
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger()
config = ConfigLoader()

class Preprocessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(
            clipLimit=config.get('preprocessing.clahe_clip_limit', 3.0),
            tileGridSize=tuple(config.get('preprocessing.clahe_tile_grid_size', [8, 8]))
        )

    def preprocess_face(self, face_roi):
        # Handle input
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.copy()

        # Gamma correction
        gamma = config.get('preprocessing.gamma', 1.1)
        gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(gray, gamma_table)

        # CLAHE
        enhanced = self.clahe.apply(gamma_corrected)
        
        # Bilateral
        filtered = cv2.bilateralFilter(enhanced, 9, 
                                      config.get('preprocessing.bilateral_sigma_d', 75),
                                      config.get('preprocessing.bilateral_sigma_r', 75))
        
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoising(filtered, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Enhanced Unsharp masking (stronger alpha for more sharpness)
        sigma = config.get('preprocessing.unsharp_sigma', 2.0)
        blurred = cv2.GaussianBlur(denoised, (0, 0), sigma)
        sharpen_strength = config.get('preprocessing.sharpening_strength', 2.0)  # New: Tune 1.5-2.5
        sharpened = cv2.addWeighted(denoised, sharpen_strength, blurred, (1 - sharpen_strength), 0)
        
        # ✅ NEW: Kernel-based sharpening (convolution for edge boost)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])  # Classic sharpening kernel
        kernel_sharpened = cv2.filter2D(sharpened, -1, kernel)
        
        # Resize (bicubic for low-res)
        h, w = kernel_sharpened.shape
        if max(h, w) < 200:
            upscaled = cv2.resize(kernel_sharpened, (200, 200), interpolation=cv2.INTER_CUBIC)
        else:
            upscaled = cv2.resize(kernel_sharpened, (200, 200))
        
        # Final sharpness check
        sharpness = cv2.Laplacian(upscaled, cv2.CV_64F).var()
        if sharpness < 60:
            logger.warning(f"Low post-sharpen sharpness: {sharpness:.1f} (consider better lighting)")
        
        logger.debug(f"Preprocess sharpness boost: {sharpness:.1f}")
        return upscaled