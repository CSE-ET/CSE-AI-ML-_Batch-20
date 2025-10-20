import cv2
from pathlib import Path
from tqdm import tqdm
from utils.logger import setup_logger
from .detection import FaceDetector

logger = setup_logger()


class UniqueCapturer:
    def __init__(self, target_count=100):
        self.detector = FaceDetector()
        self.data_dir = Path("data/train")
        self.min_quality_score = 80  # Laplacian variance threshold
        self.target_count = target_count

    def compute_quality_score(self, roi_gray):
        """Compute Laplacian variance for sharpness."""
        return cv2.Laplacian(roi_gray, cv2.CV_64F).var()

    def run_capture_session(self, subject_id):
        subject_dir = self.data_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        # ✅ Dynamic cam setup: Probe supported formats
        cap = cv2.VideoCapture(0)  # Backend auto-detects (DSHOW on Win, V4L on Linux, etc.)
        
        # Try HD first, fallback to SD
        resolutions = [(1280, 720, 30), (640, 480, 20), (320, 240, 15)]  # FPS per res
        success = False
        for w, h, fps in resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            if actual_w == w and actual_h == h and actual_fps >= fps * 0.8:  # 80% tolerance
                logger.info(f"✅ Cam optimized: {w}x{h} @ {actual_fps:.1f} FPS")
                success = True
                break
        
        if not success:
            logger.warning("Fallback to lowest res—check cam connection.")

        # ✅ App-only exposure tweak (no system change)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Semi-auto: Balances without flicker
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 1.2)  # +20% brightness (0-1 range, portable)
        cap.set(cv2.CAP_PROP_CONTRAST, 1.1)  # +10% contrast

        captured_count = 0
        pbar = tqdm(total=self.target_count, desc=f"Capturing {subject_id}")

        while captured_count < self.target_count:
            ret, frame = cap.read()
            if not ret:
                continue

        cap.release()
        cv2.destroyAllWindows()
        pbar.close()
        logger.info("✅ Capture session completed!")