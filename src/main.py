import argparse
import numpy as np
from src.detection import FaceDetector
from src.preprocessing import Preprocessor
from src.training import Trainer
from src.recognition import Recognizer
from src.gui import GUI
from src.capture import UniqueCapturer  # Added for capture mode
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader


logger = setup_logger()
config = ConfigLoader()

def main():
    parser = argparse.ArgumentParser(description="Real-Time Facial Recognition System")
    parser.add_argument('--mode', choices=['train', 'recognize', 'capture'], required=True,
                        help="Mode: train, recognize, or capture")
    args = parser.parse_args()

    if args.mode == 'train':
        trainer = Trainer()
        faces, labels = trainer.load_dataset()
        trainer.train(faces, labels)
        logger.info("Training completed.")

    elif args.mode == 'recognize':
        trainer = Trainer()
        if not trainer.model_path.exists():
            raise FileNotFoundError("Train model first: python src/main.py --mode train")
        trainer.model.read(str(trainer.model_path))  # Load model
        
        # Load sample training faces for threshold computation (reuse from dataset)
        sample_faces, _ = trainer.load_dataset()
        tau = Recognizer(trainer).compute_adaptive_threshold(sample_faces)
        
        detector = FaceDetector()
        preprocessor = Preprocessor()
        recognizer = Recognizer(trainer)
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

if __name__ == "__main__":
    main()