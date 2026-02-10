import cv2
import time
import os
from utils.logger import setup_logger
from utils.notifications import notify_unknown_face
from utils import db

logger = setup_logger()

class GUI:
    def __init__(self, recognizer, detector, preprocessor, tau, use_db=False):
        self.recognizer = recognizer
        self.detector = detector
        self.preprocessor = preprocessor
        self.tau = tau
        self.use_db = use_db
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2
        self.unknown_face_cooldown = {}  # Prevent spam notifications for same face

    def run(self, webcam_id=0):
        # Try default open, then Windows-specific backends if needed
        cap = cv2.VideoCapture(webcam_id)
        if not cap.isOpened():
            try:
                cap = cv2.VideoCapture(webcam_id, cv2.CAP_DSHOW)
            except Exception:
                pass
        if not cap.isOpened():
            try:
                cap = cv2.VideoCapture(webcam_id, cv2.CAP_MSMF)
            except Exception:
                pass
        if not cap.isOpened():
            logger.error(f"Cannot open webcam (id={webcam_id}). Check device, permissions, or try another index.")
            print("ERROR: Cannot open webcam. See logs for details.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)

        fps_start = time.time()
        frame_count = 0
        unknown_count = 0  # Track unknown face detections for de-duplication

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed from webcam; retrying...")
                # Small sleep to avoid busy loop and allow camera to recover
                time.sleep(0.1)
                continue

            frame_count += 1
            faces = self.detector.detect_faces(frame)

            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                preprocessed = self.preprocessor.preprocess_face(face_roi)
                identity, conf = self.recognizer.recognize_face(preprocessed, self.tau)

                # Color-coding
                if identity == "Unknown":
                    color = (0, 0, 255)  # Red
                    label_text = f"Unknown ({conf:.2f})"
                    
                    # Capture and store unknown face
                    unknown_count += 1
                    self._handle_unknown_face(face_roi, unknown_count)
                else:
                    color = (0, 255, 0)  # Green
                    label_text = f"{identity} ({conf:.2f})"

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, self.thickness)
                cv2.putText(frame, label_text, (x, y-10), self.font, self.font_scale, color, self.thickness)

            # FPS display every 30 frames
            if frame_count % 30 == 0:
                fps = frame_count / (time.time() - fps_start)
                logger.info(f"FPS: {fps:.2f}")
                frame_count = 0
                fps_start = time.time()

            cv2.imshow('Real-Time Facial Recognition', frame)

            # Quit manually
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _handle_unknown_face(self, face_roi, unknown_id):
        """
        Capture unknown face, store in DB, and send notification.
        De-duplicates notifications using cooldown.
        """
        current_time = time.time()
        cooldown_key = "unknown_notification"
        
        # Only notify once every 5 seconds to avoid spam
        if cooldown_key not in self.unknown_face_cooldown or (current_time - self.unknown_face_cooldown[cooldown_key]) > 5:
            logger.warning("Unknown face detected! Storing in database and sending alert...")
            notify_unknown_face()
            self.unknown_face_cooldown[cooldown_key] = current_time
        
        # Always store the face in DB (for forensics)
        if self.use_db:
            try:
                # Encode face as JPEG BLOB
                success, face_jpeg = cv2.imencode('.jpg', face_roi)
                if success:
                    timestamp = int(current_time * 1000)
                    filename = f"unknown_{timestamp}_{unknown_id}.jpg"
                    db.add_image("unknown", face_jpeg.tobytes(), is_train=False, filename=filename)
                    logger.info(f"Unknown face stored in DB: {filename}")
            except Exception as e:
                logger.error(f"Failed to store unknown face in DB: {e}")
                # Fallback: try to save to filesystem
                try:
                    if not os.path.exists("data/unknown"):
                        os.makedirs("data/unknown")
                    timestamp = int(current_time * 1000)
                    filepath = f"data/unknown/unknown_{timestamp}_{unknown_id}.jpg"
                    cv2.imwrite(filepath, face_roi)
                    logger.info(f"Unknown face saved to filesystem: {filepath}")
                except Exception as fs_error:
                    logger.error(f"Failed to save unknown face to filesystem: {fs_error}")
