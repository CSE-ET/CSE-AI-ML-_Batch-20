import cv2
import numpy as np
import pickle
from pathlib import Path
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader
import os
from .preprocessing import Preprocessor


logger = setup_logger()
config = ConfigLoader()

class Trainer:
    
    def __init__(self, model_path="models/lbph_model.yaml"):
        self.preprocessor = Preprocessor()
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = cv2.face.LBPHFaceRecognizer_create(
            radius=config.get('training.lbph_radius', 2),
            neighbors=config.get('training.lbph_neighbors', 8),
            grid_x=config.get('training.lbph_grid_x', 8),
            grid_y=config.get('training.lbph_grid_y', 8)
        )

        self.subjects_db_path = "models/labels.pkl"
        self.face_size = config.get('preprocessing.face_size', 200)
        self.min_sharpness = config.get('preprocessing.min_sharpness_variance', 100)


    def load_dataset(self, data_dir="data/train"):
        faces = []
        labels = []
        label_map = {}
        current_id = 0

        for subject_dir in sorted(Path(data_dir).iterdir()):
            if not subject_dir.is_dir():
                continue

            label = subject_dir.name
            label_map[label] = current_id

            for img_path in subject_dir.glob("*.jpg"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Apply full preprocessing (mimics inference)
                processed_img = self.preprocessor.preprocess_face(img)  # Now handles gray

                # Sharpness check on processed image (more robust)
                sharpness = cv2.Laplacian(processed_img, cv2.CV_64F).var()
                if sharpness < self.min_sharpness:
                    logger.warning(f"Low sharpness image skipped: {img_path} (variance {sharpness:.2f})")
                    continue

                faces.append(processed_img)
                labels.append(current_id)

            current_id += 1

        self.save_label_map(label_map)
        logger.info(f"Loaded {len(faces)} processed images for {len(set(labels))} subjects.")
        return faces, labels


    def save_label_map(self, label_map):
        with open(self.subjects_db_path, 'wb') as f:
            pickle.dump(label_map, f)

    def load_label_map(self):
        if not os.path.exists(self.subjects_db_path):
            return {}
        with open(self.subjects_db_path, 'rb') as f:
            return pickle.load(f)

    def train(self):
        faces, labels = self.load_dataset()
        if not faces:
            print("No valid images found for training!")
            return

        print(f"Starting training on {len(faces)} images for {len(set(labels))} subjects...")
        self.model.train(faces, np.array(labels))
        self.model.save(str(self.model_path))
        print("LBPH Training completed!")
        print(f"Model saved at: {self.model_path}")

    def incremental_update(self, new_face, new_label):
        self.model.update([new_face], np.array([new_label]))
        self.model.save(str(self.model_path))
        print(f"Model updated with new label {new_label}")
