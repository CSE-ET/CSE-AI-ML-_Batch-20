#!/usr/bin/env python3
"""
Test script to verify that unknown faces are properly rejected.
This script:
1. Tests recognition on known and unknown faces
2. Verifies that unknown faces are rejected with confidence above threshold
3. Prints a summary report
"""
import cv2
import numpy as np
from pathlib import Path
from src.training import Trainer
from src.recognition import Recognizer
from utils.logger import setup_logger

logger = setup_logger()

def test_recognition():
    """Test recognition on known and unknown faces."""
    
    logger.info("=" * 60)
    logger.info("RECOGNITION TEST")
    logger.info("=" * 60)
    
    # Initialize trainer and recognizer
    trainer = Trainer()
    recognizer = Recognizer(trainer)
    
    # Compute adaptive threshold
    faces, labels = trainer.load_dataset()
    recognizer.compute_adaptive_threshold(faces)
    
    logger.info(f"Model loaded with {len(set(labels))} known subjects")
    logger.info(f"Recognition threshold: {recognizer.tau:.2f}")
    logger.info(f"Label map: {recognizer.label_map}")
    logger.info("")
    
    # Test on known faces
    logger.info("Testing on KNOWN faces (should be recognized):")
    logger.info("-" * 60)
    
    test_dir = Path("data/test")
    if test_dir.exists():
        known_correct = 0
        known_total = 0
        
        for subject_dir in sorted(test_dir.iterdir()):
            if not subject_dir.is_dir() or subject_dir.name.lower() == "unknown":
                continue
            
            # Try first image in each known subject's test folder
            img_files = list(subject_dir.glob("*.jpg"))
            if img_files:
                img_path = img_files[0]
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    processed = recognizer.preprocess_face(img)
                    result_name, conf = recognizer.recognize_face(processed, recognizer.tau)
                    expected_name = subject_dir.name.lower()
                    
                    is_correct = result_name.lower() == expected_name
                    known_total += 1
                    if is_correct:
                        known_correct += 1
                    
                    status = "✓ CORRECT" if is_correct else "✗ WRONG"
                    logger.info(f"{status}: Expected '{expected_name}', Got '{result_name}' (conf={conf:.2f})")
        
        if known_total > 0:
            logger.info(f"\nKnown faces accuracy: {known_correct}/{known_total} ({100*known_correct/known_total:.1f}%)")
    
    logger.info("")
    
    # Test on unknown faces
    logger.info("Testing on UNKNOWN faces (should be rejected):")
    logger.info("-" * 60)
    
    if (test_dir / "unknown").exists():
        unknown_correct = 0
        unknown_total = 0
        
        for img_file in (test_dir / "unknown").glob("*.jpg"):
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                processed = recognizer.preprocess_face(img)
                result_name, conf = recognizer.recognize_face(processed, recognizer.tau)
                
                is_rejected = result_name == "Unknown"
                unknown_total += 1
                if is_rejected:
                    unknown_correct += 1
                
                status = "✓ REJECTED" if is_rejected else "✗ FALSE POSITIVE"
                logger.info(f"{status}: Got '{result_name}' (conf={conf:.2f})")
        
        if unknown_total > 0:
            logger.info(f"\nUnknown faces rejection rate: {unknown_correct}/{unknown_total} ({100*unknown_correct/unknown_total:.1f}%)")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST COMPLETE!")
    logger.info("=" * 60)

if __name__ == "__main__":
    test_recognition()
