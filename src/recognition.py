import cv2
import numpy as np
from collections import deque
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger()
config = ConfigLoader()

class Recognizer:
    def __init__(self, trainer):  # ✅ Accepts 'trainer' arg—fixes TypeError
        self.trainer = trainer
        self.model = trainer.model
        self.label_map = trainer.load_label_map()
        self.reverse_map = {v: k for k, v in self.label_map.items()}
        self.conf_history = deque(maxlen=5)  # Temporal smoothing
        self.tau = config.get('recognition.default_threshold', 100)  # LBPH typical threshold

    def preprocess_face(self, face):
        """Minimal fallback; now redundant since input is preprocessed."""
        # Assuming input is already gray 200x200; no-op if so
        if len(face.shape) == 2 and face.shape[0:2] == (200, 200):
            return face
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        return cv2.resize(gray, (200, 200))

    def compute_adaptive_threshold(self, training_faces):
        """Compute on preprocessed samples (no extra preprocess needed now)."""
        if not training_faces:
            logger.warning("No training faces for threshold computation; using default.")
            return self.tau

        predictions = []
        for face in training_faces[:min(10, len(training_faces))]:
            # Skip redundant preprocess_face; use face directly (already processed)
            _, conf = self.model.predict(face)
            predictions.append(conf)
        
        if not predictions:
            return self.tau
            
        mu = np.mean(predictions)
        sigma = np.std(predictions)
        tau = mu + 2.0 * sigma  # Increased multiplier for safety (LBPH distances ~50-150)
        tau = max(80, min(tau, 200))  # Clamp to reasonable LBPH range
        self.tau = tau
        logger.info(f"Adaptive threshold: {tau:.2f} (mu={mu:.2f}, sigma={sigma:.2f})")
        return tau

    def recognize_face(self, face, tau):
        # Remove redundant preprocess; assume input is preprocessed
        label_id, confidence = self.model.predict(face)
        self.conf_history.append(confidence)
        smoothed_conf = np.mean(self.conf_history)

        if smoothed_conf < tau:
            name = self.reverse_map.get(label_id, "Unknown")
            logger.debug(f"Match: {name} (conf={smoothed_conf:.2f} < {tau:.2f})")
            return name, smoothed_conf
        else:
            logger.debug(f"No match (conf={smoothed_conf:.2f} >= {tau:.2f})")
            return "Unknown", smoothed_conf

    def enroll_unknown(self, face, label="New_User"):
        """Add a new face to the model."""
        preprocessed = self.preprocess_face(face)
        new_id = len(self.label_map)
        self.label_map[label] = new_id
        self.trainer.incremental_update(preprocessed, new_id)
        self.reverse_map[new_id] = label
        logger.info(f"Enrolled new subject: {label}")

    def save_label_map(self):
        self.trainer.save_label_map(self.label_map)