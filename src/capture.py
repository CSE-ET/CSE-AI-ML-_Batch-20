import cv2
from pathlib import Path
from tqdm import tqdm
from utils.logger import setup_logger
from .detection import FaceDetector
from utils import db as db_helper
import cv2

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
        # Ensure DB exists
        db_helper.init_db()

        subject_dir = self.data_dir / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        # ✅ Dynamic cam setup: Probe supported formats
        # Try to open camera with common backends; on Windows prefer DSHOW/MSMF
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            try:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            except Exception:
                pass
        if not cap.isOpened():
            try:
                cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
            except Exception:
                pass

        if not cap.isOpened():
            logger.error("Unable to open webcam (index 0). Check permissions, other apps, or try another index.")
            return 0

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
                logger.info(f"Cam optimized: {w}x{h} @ {actual_fps:.1f} FPS")
                success = True
                break

        if not success:
            logger.warning("Fallback to lowest res — check cam connection.")

        # ✅ App-only exposure tweak (no system change)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Semi-auto: Balances without flicker
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 1.2)  # +20% brightness (0–1 range, portable)
        cap.set(cv2.CAP_PROP_CONTRAST, 1.1)  # +10% contrast

        captured_count = 0
        pbar = tqdm(total=self.target_count, desc=f"Capturing {subject_id}")

        while captured_count < self.target_count:
            ret, frame = cap.read()
            if not ret:
                continue

            # ✅ Detect faces in the frame
            faces = self.detector.detect_faces(frame)
            for (x, y, w, h) in faces:
                roi = frame[y:y + h, x:x + w]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                quality = self.compute_quality_score(gray)

                # ✅ Only save sharp, clear images
                # After quality check
                if quality >= self.min_quality_score:
                    # ✅ NEW: Face validation (reject small/blurry/non-square ROIs)
                    if w < 100 or h < 100 or abs(w - h) / max(w, h) > 0.2:  # Min size 100px, aspect <20% off square
                        logger.warning(f"Invalid face ROI rejected: size {w}x{h}, quality {quality:.1f}")
                        continue
                    # Save into DB as JPEG bytes (preferred)
                    success, buffer = cv2.imencode('.jpg', gray, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if success:
                        img_bytes = buffer.tobytes()
                        try:
                            db_helper.add_image(subject_id, img_bytes, is_train=1, filename=f"img{captured_count + 1:03d}.jpg")
                        except Exception:
                            # fallback to file save if DB insert fails
                            img_path = subject_dir / f"img{captured_count + 1:03d}.jpg"
                            cv2.imwrite(str(img_path), gray, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    else:
                        img_path = subject_dir / f"img{captured_count + 1:03d}.jpg"
                        cv2.imwrite(str(img_path), gray, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    captured_count += 1
                    pbar.update(1)
                    logger.info(f"Captured {captured_count}/{self.target_count} for {subject_id} (score: {quality:.1f}, size: {w}x{h})")
            # ✅ Draw rectangle around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow(f"Capturing {subject_id} - Press 'q' to stop", frame)

            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        pbar.close()
        logger.info("Capture session completed!")
        return captured_count