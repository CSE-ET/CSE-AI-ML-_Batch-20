import cv2
import numpy as np
from collections import deque
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

logger = setup_logger()
config = ConfigLoader()

class Recognizer:
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.label_map = trainer.load_label_map()
        if not self.label_map:
            logger.warning("Empty label map—train first!")
            self.label_map = {}
        self.reverse_map = {v: k for k, v in self.label_map.items()}
        self.conf_history = deque(maxlen=10)  # Increased from 5 to 10 for better stability
        self.tau = config.get('recognition.default_threshold', 70)  # More relaxed threshold
        self.strict_threshold = config.get('recognition.strict_threshold', 50)  # Reasonable strict threshold

    def preprocess_face(self, face):
        if len(face.shape) == 2 and face.shape[:2] == (200, 200):
            return face
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        return cv2.resize(gray, (200, 200))

    def compute_adaptive_threshold(self, training_faces):
        if not training_faces:
            logger.warning("No training faces for threshold computation; using default.")
            return self.tau

        predictions = []
        num_samples = min(20, len(training_faces))
        for face in training_faces[:num_samples]:
            _, conf = self.model.predict(face)
            predictions.append(conf)

        if not predictions:
            return self.tau
        
        known_confs = [conf for conf in predictions if conf < 50]  # Low conf = known class samples
        if known_confs:
            mu = np.mean(known_confs)
            sigma = np.std(known_confs)
            # Use higher multiplier for better acceptance of known faces
            tau = mu + 2.0 * sigma  # Higher multiplier for reasonable threshold
            log_mu, log_sigma = mu, sigma
            logger.info(f"Using known-weighted tau (n_known={len(known_confs)})")
        else:
            mu = np.mean(predictions)
            sigma = np.std(predictions)
            tau = mu + 1.5 * sigma  # Use reasonable multiplier
            log_mu, log_sigma = mu, sigma
            logger.info(f"Using full tau (no known confs detected)")

        # Relaxed clamp: 40-80 (allows better recognition of known faces)
        tau = max(40, min(tau, 80))
        self.tau = tau
        logger.info(f"Adaptive threshold set: tau={tau:.2f} (mu={log_mu:.2f}, sigma={log_sigma:.2f}, n={num_samples})")
        return tau

    def recognize_face(self, face, tau):
        label_id, confidence = self.model.predict(face)
        self.conf_history.append(confidence)
        smoothed_conf = np.mean(self.conf_history) if self.conf_history else confidence

        # Decision logic: stricter acceptance criteria
        candidate_name = self.reverse_map.get(label_id, "Unknown")
        
        # Accept as known ONLY if confidence is strictly below threshold
        if smoothed_conf < tau:
            # Extra check: is it really confident enough?
            if smoothed_conf < self.strict_threshold:
                name = candidate_name  # Very confident match
            else:
                # Moderate confidence - use a margin check
                margin = abs(smoothed_conf - tau)
                if margin > 15:  # Need at least 15 points of headroom
                    name = candidate_name
                else:
                    name = "Unknown"  # Too close to threshold
        else:
            # Confidence exceeds threshold - reject as unknown
            name = "Unknown"

        # Additional safety: verify not misclassified by checking all predictions
        if name != "Unknown" and len(self.reverse_map) > 1:
            # Double-check: smoothed confidence should be significantly better than next-best candidate
            # If it's not, this might be a false positive
            if smoothed_conf > 40:  # If confidence is already high (weak match), be extra careful
                name = "Unknown"
                logger.debug(f"Confidence too high (weak match): {smoothed_conf:.2f}")

        if name != "Unknown":
            logger.info(f"Recognition result → {name} (conf={smoothed_conf:.2f}, tau={tau:.2f})")
        else:
            logger.info(f"Rejected as Unknown (conf={smoothed_conf:.2f}, tau={tau:.2f})")
        
        return name, smoothed_conf

    def enroll_unknown(self, face, label="New_User"):
        preprocessed = self.preprocess_face(face)
        new_id = len(self.label_map)
        self.label_map[label] = new_id
        self.trainer.incremental_update(preprocessed, new_id)
        self.reverse_map[new_id] = label
        logger.info(f"Enrolled new subject: {label}")

    def save_label_map(self):
        self.trainer.save_label_map(self.label_map)