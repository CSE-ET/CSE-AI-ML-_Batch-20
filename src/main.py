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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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

    args = parser.parse_args()

    if args.mode == 'train':
        trainer = Trainer()
        faces, labels = trainer.load_dataset()
        trainer.train()
        logger.info("Training completed.")

    elif args.mode == 'recognize':
        trainer = Trainer()
        if not trainer.model_path.exists():
            raise FileNotFoundError("Train model first: python src/main.py --mode train")
        trainer.model.read(str(trainer.model_path))  # Load model
        
        sample_faces, _ = trainer.load_dataset()
        
        detector = FaceDetector()
        preprocessor = Preprocessor()
        recognizer = Recognizer(trainer)
        tau = recognizer.compute_adaptive_threshold(sample_faces)
        
        gui = GUI(recognizer, detector, preprocessor, tau)
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
