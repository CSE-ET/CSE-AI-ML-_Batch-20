import argparse
import numpy as np
from .detection import FaceDetector
from .preprocessing import Preprocessor
from .training import Trainer
from .recognition import Recognizer
from .gui import GUI
from .capture import UniqueCapturer
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader
import matplotlib.pyplot as plt
from collections import Counter
import json

logger = setup_logger()
config = ConfigLoader()

def main():
    parser = argparse.ArgumentParser(description="Real-Time Facial Recognition System")
    parser.add_argument('--mode', choices=['train', 'recognize', 'capture', 'evaluate'], required=True,
                        help="Mode: train, recognize, capture, or evaluate")
    parser.add_argument('--test_dir', default='data/test', help="Path to test dataset (for evaluate mode)")
    parser.add_argument('--use_db', action='store_true', help="Load images from the project's SQLite DB instead of filesystem")

    args = parser.parse_args()

    if args.mode == 'train':
        trainer = Trainer()
        faces, labels = trainer.load_dataset(use_db=args.use_db)
        trainer.train()
        logger.info("Training completed.")

    elif args.mode == 'recognize':
        trainer = Trainer()
        if not trainer.model_path.exists():
            raise FileNotFoundError("Train model first: python src/main.py --mode train")
        trainer.model.read(str(trainer.model_path))  # Load model
        
        sample_faces, _ = trainer.load_dataset(use_db=args.use_db)
        
        detector = FaceDetector()
        preprocessor = Preprocessor()
        recognizer = Recognizer(trainer)
        tau = recognizer.compute_adaptive_threshold(sample_faces)
        
        gui = GUI(recognizer, detector, preprocessor, tau, use_db=args.use_db)
        gui.run(config.get('recognition.webcam_id', 0))

    elif args.mode == 'capture':
        capturer = UniqueCapturer()
        subject_id = input("Enter subject ID (e.g., hasini): ").strip()
        if not subject_id:
            logger.error("Subject ID required.")
            return
        capturer.run_capture_session(subject_id)
        logger.info("Capture session completed.")

    elif args.mode == 'evaluate':
        trainer = Trainer()
        if not trainer.model_path.exists():
            raise FileNotFoundError("Train model first: python src/main.py --mode train")
        trainer.model.read(str(trainer.model_path))
        
        # Load test dataset
        if args.use_db:
            test_faces, test_labels = trainer.load_dataset(use_db=True)
        else:
            test_faces, test_labels = trainer.load_dataset(args.test_dir)
        if len(test_faces) == 0:
            raise ValueError(f"No test images in {args.test_dir}. Capture some first!")
        
        recognizer = Recognizer(trainer)
        tau = recognizer.compute_adaptive_threshold(test_faces[:10])
        logger.info(f"Using tau={tau:.2f} for evaluation on {len(test_faces)} test images.")
        
        # Predict each face
        predictions = []
        for face in test_faces:
            name, conf = recognizer.recognize_face(face, tau)
            predictions.append(name)
        
        # Map numeric test labels to names
        true_names = [recognizer.reverse_map.get(lbl, "Unknown") for lbl in test_labels]
        
        # Metrics
        # Metrics
        try:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        except Exception:
            def accuracy_score(y_true, y_pred):
                return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true) if y_true else 0.0

            def precision_recall_fscore_support(y_true, y_pred, average=None):
                labels = list(set(y_true) | set(y_pred))
                tp = {l: 0 for l in labels}
                pred_count = {l: 0 for l in labels}
                true_count = {l: 0 for l in labels}
                for t, p in zip(y_true, y_pred):
                    true_count[t] += 1
                    pred_count[p] += 1
                    if t == p:
                        tp[t] += 1
                precisions = []
                recalls = []
                supports = []
                for l in labels:
                    p = tp[l] / pred_count[l] if pred_count[l] > 0 else 0.0
                    r = tp[l] / true_count[l] if true_count[l] > 0 else 0.0
                    precisions.append(p)
                    recalls.append(r)
                    supports.append(true_count[l])
                if average == 'weighted':
                    total = sum(supports)
                    prec = sum(p * s for p, s in zip(precisions, supports)) / total if total else 0.0
                    rec = sum(r * s for r, s in zip(recalls, supports)) / total if total else 0.0
                    f1s = []
                    for p, r in zip(precisions, recalls):
                        f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
                    f1 = sum(f * s for f, s in zip(f1s, supports)) / total if total else 0.0
                    return prec, rec, f1, None
                return precisions, recalls, None, None

        acc = accuracy_score(true_names, predictions)
        prec, rec, f1, _ = precision_recall_fscore_support(true_names, predictions, average='weighted')
        logger.info(f"Evaluation Results (tau={tau}): Accuracy={acc*100:.2f}%, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
        # Save plot
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        values = [acc, prec, rec, f1]
        plt.bar(metrics, values)
        plt.ylim(0, 1)
        plt.title('Facial Recognition Metrics')
        plt.savefig('models/eval_plot.png')
        plt.close()
        logger.info("Plot saved to models/eval_plot.png")
        
        # Save counts
        true_counts = Counter(true_names)
        pred_counts = Counter(predictions)
        logger.info(f"True label dist: {dict(true_counts)}")
        logger.info(f"Pred label dist: {dict(pred_counts)}")
        
        # Save results
        results = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
        with open('models/eval_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to models/eval_results.json")
        
        # Print summary
        print("\n=== EVALUATION SUMMARY ===")
        for key, val in results.items():
            print(f"{key.capitalize()}: {val}")
        print("==========================")

if __name__ == "__main__":
    main()
