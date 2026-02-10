import cv2
import numpy as np
import pickle
from pathlib import Path
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader
import os
from .preprocessing import Preprocessor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # Add if not there; pip install scikit-learn (or use numpy for basics)


logger = setup_logger()
config = ConfigLoader()

class Trainer:
    
    def __init__(self, model_path="models/lbph_model.yaml"):
        self.preprocessor = Preprocessor()
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Optimized LBPH parameters to reduce bias
        # Lower radius/neighbors = more uniform weighting
        # Higher grid resolution = better feature distribution
        self.model = cv2.face.LBPHFaceRecognizer_create(
            radius=config.get('training.lbph_radius', 1),      # Reduced from 2 for less bias
            neighbors=config.get('training.lbph_neighbors', 8), # Balanced
            grid_x=config.get('training.lbph_grid_x', 10),     # Increased from 8 for better distribution
            grid_y=config.get('training.lbph_grid_y', 10)      # Increased from 8 for better distribution
        )

        # Load existing model if available
        if self.model_path.exists():
            logger.info(f"Loading existing model from {self.model_path}")
            self.model.read(str(self.model_path))

        self.subjects_db_path = "models/labels.pkl"
        self.face_size = config.get('preprocessing.face_size', 200)
        self.min_sharpness = config.get('preprocessing.min_sharpness_variance', 100)


    def load_dataset(self, data_dir="data/train", use_db=False, balance_dataset=True):
        """
        Load dataset either from filesystem (`data_dir`) or from DB if `use_db=True`.
        Returns (faces, labels) where faces are numpy arrays (preprocessed) and labels are numeric IDs.
        Excludes "unknown" subject from training (used for testing only).
        
        Args:
            balance_dataset: If True, balances dataset to equal images per subject (fixes LBPH bias)
        """
        faces = []
        labels = []
        label_map = {}
        current_id = 0
        subject_images = {}  # Track images per subject for balancing

        if use_db:
            # Lazy import of DB helper to avoid circulars at import time
            from utils import db as db_helper
            rows = db_helper.get_images(is_train=True)
            # rows: list of (image_bytes, subject_name)
            for img_bytes, subject_name in rows:
                # SKIP "unknown" subject - it's not a known identity, only for testing/forensics
                if subject_name.lower() == "unknown":
                    logger.debug(f"Skipping 'unknown' subject image during training")
                    continue
                
                # Normalize subject names to lowercase for consistency
                subject_name = subject_name.lower()
                
                if subject_name not in label_map:
                    label_map[subject_name] = current_id
                    current_id += 1

                # Decode bytes to image
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                processed_img = self.preprocessor.preprocess_face(img)
                sharpness = cv2.Laplacian(processed_img, cv2.CV_64F).var()
                if sharpness < self.min_sharpness:
                    logger.warning(f"Low sharpness image skipped (DB image) for {subject_name} (variance {sharpness:.2f})")
                    continue

                faces.append(processed_img)
                subject_id = label_map[subject_name]
                labels.append(subject_id)
                
                # Track for balancing
                if subject_id not in subject_images:
                    subject_images[subject_id] = []
                subject_images[subject_id].append(len(faces) - 1)

            # **FIX BIAS**: Balance dataset to equal images per subject
            if balance_dataset and subject_images:
                faces, labels = self._balance_dataset(faces, labels, subject_images)
                logger.info(f"Dataset balanced to prevent LBPH bias")

            self.save_label_map(label_map)
            logger.info(f"Loaded {len(faces)} processed images for {len(set(labels))} subjects (DB). Excluded 'unknown' subject.")
            return faces, labels

        # Filesystem fallback (existing behavior)
        for subject_dir in sorted(Path(data_dir).iterdir()):
            if not subject_dir.is_dir():
                continue

            label = subject_dir.name
            # SKIP "unknown" subject - it's not a known identity, only for testing/forensics
            if label.lower() == "unknown":
                logger.debug(f"Skipping 'unknown' subject directory during training")
                continue
            
            # Normalize subject names to lowercase for consistency
            label = label.lower()
            
            if label not in label_map:
                label_map[label] = current_id
                current_id += 1

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

                subject_id = current_id
                faces.append(processed_img)
                labels.append(subject_id)
                
                # Track for balancing
                if subject_id not in subject_images:
                    subject_images[subject_id] = []
                subject_images[subject_id].append(len(faces) - 1)

            current_id += 1

        # **FIX BIAS**: Balance dataset to equal images per subject
        if balance_dataset and subject_images:
            faces, labels = self._balance_dataset(faces, labels, subject_images)
            logger.info(f"Dataset balanced to prevent LBPH bias")

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
        # Load from both filesystem AND database, combining all available training data
        faces, labels = self.load_dataset(use_db=True)
        if not faces:
            print("No valid images found for training!")
            return

        # **FIX BIAS**: Shuffle training data to prevent LBPH from biasing toward last subjects
        # LBPH gives more weight to later training samples, so randomization is crucial
        shuffled_indices = np.random.permutation(len(faces))
        faces = [faces[i] for i in shuffled_indices]
        labels = [labels[i] for i in shuffled_indices]
        
        logger.info(f"Shuffled training data to prevent LBPH bias toward last subjects")
        print(f"Starting training on {len(faces)} images for {len(set(labels))} subjects...")
        logger.info(f"Training data distribution: {dict(zip(*np.unique(labels, return_counts=True)))}") 
        
        self.model.train(faces, np.array(labels))
        self.model.save(str(self.model_path))
        print("LBPH Training completed!")
        print(f"Model saved at: {self.model_path}")
        logger.info(f"Model saved at: {self.model_path}")
    
    def _balance_dataset(self, faces, labels, subject_images):
        """
        Balance dataset to have equal images per subject.
        This prevents LBPH from biasing toward subjects with more training images.
        """
        # Find minimum and maximum images per subject
        image_counts = [len(indices) for indices in subject_images.values()]
        min_count = min(image_counts)
        max_count = max(image_counts)
        
        logger.info(f"Dataset imbalance - Min: {min_count}, Max: {max_count} images per subject")
        
        # Sample equal number of images from each subject
        balanced_indices = []
        for subject_id, indices in subject_images.items():
            # Take random sample of min_count images from this subject
            sampled = np.random.choice(indices, size=min_count, replace=False)
            balanced_indices.extend(sampled)
        
        # Reorder faces and labels
        balanced_indices = sorted(balanced_indices)
        balanced_faces = [faces[i] for i in balanced_indices]
        balanced_labels = [labels[i] for i in balanced_indices]
        
        logger.info(f"Balanced dataset: {len(balanced_faces)} images ({min_count} per subject)")
        return balanced_faces, balanced_labels
    
    def evaluate(self, test_faces, test_labels, recognizer, tau=80):
        predictions = []
        confidences = []
        for face, true_label in zip(test_faces, test_labels):
            pred_label, conf = recognizer.recognize_face(face, tau)
            predictions.append(pred_label)  # String label for metrics
            confidences.append(conf)
        
        # Map strings to IDs for numeric metrics
        pred_ids = [list(recognizer.reverse_map.keys())[list(recognizer.reverse_map.values()).index(p)] if p != "Unknown" else -1 for p in predictions]
        true_ids = test_labels  # Assume numeric IDs
        
        acc = accuracy_score(true_ids, pred_ids)
        prec, rec, f1, _ = precision_recall_fscore_support(true_ids, pred_ids, average='weighted', zero_division=0)
        
        logger.info(f"Evaluation Results (tau={tau}): Accuracy={acc:.2%}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, Avg Conf={np.mean(confidences):.2f}")
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'avg_conf': np.mean(confidences)}

    def incremental_update(self, new_face, new_label):
        self.model.update([new_face], np.array([new_label]))
        self.model.save(str(self.model_path))
        print(f"Model updated with new label {new_label}")
