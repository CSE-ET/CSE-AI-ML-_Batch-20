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
        self.reverse_map = {v: k for k, v in self.label_map.items()}
        self.k = config.get('recognition.confidence_threshold_k', 1.5)
        self.alpha = config.get('recognition.temporal_smoothing_alpha', 0.7)
        self.conf_history = deque(maxlen=5)  # Temporal smoothing

    def compute_adaptive_threshold(self, training_faces):
        # Compute μ and σ from training matches (simplified: use sample predictions)
        predictions = []
        for face in training_faces[:10]:  # Sample
            _, conf = self.model.predict(face)
            predictions.append(conf)
        mu = np.mean(predictions)
        sigma = np.std(predictions)
        tau = mu - self.k * sigma
        logger.info(f"Adaptive threshold tau: {tau:.2f} (mu={mu:.2f}, sigma={sigma:.2f})")
        return tau

    def recognize_face(self, preprocessed_face, tau):
        label_id, confidence = self.model.predict(preprocessed_face)
        
        # Temporal smoothing
        self.conf_history.append(confidence)
        smoothed_conf = np.mean(self.conf_history)
        
        if smoothed_conf > tau:
            return self.reverse_map.get(label_id, "Unknown"), smoothed_conf
        else:
            return "Unknown", smoothed_conf  # Reject unknown

    def enroll_unknown(self, face, label="New_User"):
        # Optional k-means for clustering (simplified: direct enroll)
        new_id = len(self.label_map)
        self.label_map[label] = new_id
        self.trainer.incremental_update(face, new_id)
        self.reverse_map[new_id] = label
        logger.info(f"Enrolled new subject: {label}")