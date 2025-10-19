import cv2
from pathlib import Path
from tqdm import tqdm
from utils.logger import setup_logger
from src.detection import FaceDetector

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
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        captured_count = 0
        pbar = tqdm(total=self.target_count, desc=f"Capturing {subject_id}")

        while captured_count < self.target_count:
            ret, frame = cap.read()
            if not ret:
                continue
            
            faces = self.detector.detect_faces(frame)
            if len(faces) == 0:
                continue  # No faces detected, skip

            # Take the first detected face
            x, y, w, h = faces[0]
            face_roi = frame[y:y+h, x:x+w]
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_gray = cv2.resize(face_gray, (200, 200))

            # Check sharpness
            quality = self.compute_quality_score(face_gray)
            if quality >= self.min_quality_score:
                img_path = subject_dir / f"img{captured_count+1:03d}.jpg"
                cv2.imwrite(str(img_path), face_gray)
                captured_count += 1
                pbar.update(1)
                logger.info(f"Captured {captured_count}/{self.target_count} for {subject_id} (score: {quality:.1f})")

            # Optional: show frame with rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow(f"Capturing {subject_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break