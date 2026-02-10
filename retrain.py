#!/usr/bin/env python3
"""
Retraining script to fix the model.
This script:
1. Deletes the old trained model
2. Consolidates subject directories (normalize names)
3. Excludes "unknown" from training
4. Retrains the model from scratch
"""
import os
import shutil
from pathlib import Path
from src.training import Trainer
from utils.logger import setup_logger

logger = setup_logger()

def consolidate_training_data():
    """Consolidate subject directories by normalizing names to lowercase."""
    train_dir = Path("data/train")
    if not train_dir.exists():
        logger.error("data/train directory not found!")
        return
    
    # Map of old names → new names (all lowercase)
    consolidation_map = {
        "Hasini": "hasini",
        "HasiniMuvva": "hasini",  # Consolidate to single "hasini"
        "kartheesha": "kartheesha",  # Already correct
        "unknown": "unknown",  # Keep for reference but won't be trained
    }
    
    logger.info("Starting data consolidation...")
    
    for subject_dir in sorted(train_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        
        old_name = subject_dir.name
        new_name = consolidation_map.get(old_name.lower(), old_name.lower())
        
        # If name needs consolidation
        if old_name != new_name:
            new_path = train_dir / new_name
            
            if new_path.exists() and new_path != subject_dir:
                # Merge into existing directory
                logger.info(f"Merging '{old_name}' into '{new_name}'")
                for img_file in subject_dir.glob("*.jpg"):
                    shutil.move(str(img_file), str(new_path / img_file.name))
                subject_dir.rmdir()
            else:
                # Rename directory
                logger.info(f"Renaming '{old_name}' to '{new_name}'")
                subject_dir.rename(new_path)
    
    logger.info("Data consolidation complete!")
    
    # Print final structure
    logger.info("Training data structure:")
    for subject_dir in sorted(train_dir.iterdir()):
        if subject_dir.is_dir():
            num_images = len(list(subject_dir.glob("*.jpg")))
            logger.info(f"  - {subject_dir.name}: {num_images} images")

def delete_old_model():
    """Delete the old trained model."""
    model_path = Path("models/lbph_model.yaml")
    subjects_db_path = Path("models/subjects_db.pkl")
    
    if model_path.exists():
        logger.info(f"Deleting old model: {model_path}")
        model_path.unlink()
    
    if subjects_db_path.exists():
        logger.info(f"Deleting old subjects DB: {subjects_db_path}")
        subjects_db_path.unlink()
    
    logger.info("Old model deleted!")

def main():
    logger.info("=" * 60)
    logger.info("RETRAINING SCRIPT")
    logger.info("=" * 60)
    
    # Step 1: Delete old model
    delete_old_model()
    
    # Step 2: Consolidate training data
    consolidate_training_data()
    
    # Step 3: Retrain model
    logger.info("Starting model retraining...")
    trainer = Trainer()
    trainer.train()
    
    # Step 4: Compute adaptive threshold
    logger.info("Computing adaptive threshold for recognition...")
    from src.recognition import Recognizer
    faces, labels = trainer.load_dataset()
    recognizer = Recognizer(trainer)
    recognizer.compute_adaptive_threshold(faces)
    
    # Step 5: Reload model to ensure it's properly loaded
    logger.info("Reloading model for verification...")
    trainer.model.read(str(trainer.model_path))
    logger.info("Model loaded successfully!")
    
    logger.info("=" * 60)
    logger.info("RETRAINING COMPLETE!")
    logger.info("The model has been retrained with:")
    logger.info("  ✓ 'unknown' subject EXCLUDED from training")
    logger.info("  ✓ Consolidated subject names (all lowercase)")
    logger.info("  ✓ Adaptive threshold computed")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
