import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging

# Reduce noisy INFO logs from the project's logger during batch runs
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger('utils.logger').setLevel(logging.WARNING)

from src.training import Trainer
from src.recognition import Recognizer
from src.preprocessing import Preprocessor
from src.detection import FaceDetector
from pathlib import Path as PPath


def adjust_brightness(img, factor):
    img = img.astype(np.float32)
    out = img * factor
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def gaussian_blur(img, k):
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), 0)


def motion_blur(img, degree=10, angle=0):
    # Create linear motion blur kernel
    M = np.zeros((degree, degree))
    M[int((degree - 1) / 2), :] = np.ones(degree)
    # Rotate kernel to given angle
    M = cv2.warpAffine(M, cv2.getRotationMatrix2D((degree / 2 - 0.5, degree / 2 - 0.5), angle, 1.0), (degree, degree))
    M = M / M.sum()
    blurred = cv2.filter2D(img, -1, M)
    return blurred


def occlude(img, kind='mask'):
    out = img.copy()
    h, w = out.shape[:2]
    if kind == 'mask':
        # Lower half occlusion (scarf/mask)
        y1 = int(h * 0.55)
        cv2.rectangle(out, (0, y1), (w, h), (0,), -1)
    elif kind == 'sunglasses':
        # Eye region occlusion
        y1 = int(h * 0.25)
        y2 = int(h * 0.45)
        cv2.rectangle(out, (int(w * 0.15), y1), (int(w * 0.85), y2), (0,), -1)
    elif kind == 'small_patch':
        # Random small occlusion
        x = int(w * 0.6)
        y = int(h * 0.6)
        cv2.rectangle(out, (x - 20, y - 10), (x + 20, y + 10), (0,), -1)
    return out


def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def shear(img, level=0.2):
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h]])
    pts2 = np.float32([[0 + w * level, 0], [w - w * level, 0], [0, h]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def jpeg_compress(img, quality=30):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', img, encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return dec


PERTURBATIONS = {
    'brightness': lambda img, lvl: adjust_brightness(img, lvl),
    'gaussian_blur': lambda img, lvl: gaussian_blur(img, lvl),
    'motion_blur': lambda img, lvl: motion_blur(img, degree=lvl),
    'mask': lambda img, lvl: occlude(img, 'mask'),
    'sunglasses': lambda img, lvl: occlude(img, 'sunglasses'),
    'rotation': lambda img, lvl: rotate(img, lvl),
    'shear': lambda img, lvl: shear(img, lvl),
    'jpeg': lambda img, lvl: jpeg_compress(img, lvl),
}


def run_tests(args):
    trainer = Trainer()
    if not trainer.model_path.exists():
        raise FileNotFoundError("Model not found. Run training first: python -m src.main --mode train")
    trainer.model.read(str(trainer.model_path))

    # Load test data
    if args.perturb_raw:
        # Load raw files from filesystem (do not use DB)
        test_paths = []
        test_label_ids = []
        label_map = trainer.load_label_map() or {}
        data_root = PPath(args.test_dir)
        if not data_root.exists():
            raise ValueError(f"No test folder at {args.test_dir}")
        for subject_dir in sorted(data_root.iterdir()):
            if not subject_dir.is_dir():
                continue
            label = subject_dir.name
            lbl_id = label_map.get(label.lower(), None)
            for img_path in subject_dir.glob("*.jpg"):
                test_paths.append(str(img_path))
                test_label_ids.append(lbl_id if lbl_id is not None else -1)
        if not test_paths:
            raise ValueError(f"No test images found in {args.test_dir}")
    else:
        # Load preprocessed faces using Trainer.load_dataset
        test_faces, test_labels = trainer.load_dataset(args.test_dir)
        if not test_faces:
            raise ValueError(f"No test images found in {args.test_dir}")

    # Optionally sample to limit runtime
    if args.max_images and args.max_images > 0:
        if args.perturb_raw:
            test_paths = test_paths[: args.max_images]
            test_label_ids = test_label_ids[: args.max_images]
        else:
            test_faces = test_faces[: args.max_images]
            test_labels = test_labels[: args.max_images]

    recognizer = Recognizer(trainer)
    preprocessor = Preprocessor()
    detector = FaceDetector()

    # Compute tau using a small sample; prefer preprocessed faces if available
    if args.perturb_raw:
        # compute tau from a few preprocessed raw samples
        sample_faces = []
        for p in (test_paths[:10] if len(test_paths) >= 10 else test_paths):
            img = cv2.imread(p)
            if img is None:
                continue
            proc = preprocessor.preprocess_face(img)
            sample_faces.append(proc)
        if not sample_faces:
            raise ValueError("Unable to prepare sample faces for tau computation")
        tau = recognizer.compute_adaptive_threshold(sample_faces[:10])
    else:
        tau = recognizer.compute_adaptive_threshold(test_faces[:10])

    # Define severity levels per perturbation
    levels = {
        'brightness': [0.4, 0.7, 1.0, 1.3, 1.6],
        'gaussian_blur': [1, 3, 5, 7],
        'motion_blur': [5, 9, 15],
        'mask': [1],
        'sunglasses': [1],
        'rotation': [-25, -10, 0, 10, 25],
        'shear': [0.0, 0.08, 0.16],
        'jpeg': [90, 70, 50, 30]
    }

    results = defaultdict(dict)

    for p_name, lvls in levels.items():
        for lvl in lvls:
            preds = []
            if args.perturb_raw:
                for img_path, lbl in zip(test_paths, test_label_ids):
                    img = cv2.imread(img_path)
                    if img is None:
                        preds.append("Unknown")
                        continue
                    # Apply perturbation to the full raw image
                    pert = PERTURBATIONS[p_name](img.copy(), lvl)
                    # Ensure image has 3 channels for detector
                    if len(pert.shape) == 2:
                        pert_for_det = cv2.cvtColor(pert, cv2.COLOR_GRAY2BGR)
                    else:
                        pert_for_det = pert
                    # Detect faces on the perturbed image
                    try:
                        faces = detector.detect_faces(pert_for_det)
                    except Exception:
                        preds.append("Unknown")
                        continue
                    if not faces:
                        # No face detected -> count as Unknown
                        preds.append("Unknown")
                        continue
                    # Choose the largest detected face
                    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                    # Add small padding but keep within image bounds
                    pad = int(0.1 * max(w, h))
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(pert.shape[1], x + w + pad)
                    y2 = min(pert.shape[0], y + h + pad)
                    face_roi = pert[y1:y2, x1:x2]
                    proc = preprocessor.preprocess_face(face_roi)
                    name, conf = recognizer.recognize_face(proc, tau)
                    preds.append(name)
            else:
                for face in test_faces:
                    # face is preprocessed 200x200 grayscale; apply perturbation
                    perturbed = PERTURBATIONS[p_name](face.copy(), lvl)
                    # Ensure size and dtype
                    perturbed = cv2.resize(perturbed, (200, 200))
                    if len(perturbed.shape) == 3:
                        perturbed = cv2.cvtColor(perturbed, cv2.COLOR_BGR2GRAY)

                    name, conf = recognizer.recognize_face(perturbed, tau)
                    preds.append(name)

            # Map numeric true labels to names using recognizer.reverse_map
            if args.perturb_raw:
                true_names = [recognizer.reverse_map.get(lbl, "Unknown") for lbl in test_label_ids]
            else:
                true_names = [recognizer.reverse_map.get(lbl, "Unknown") for lbl in test_labels]

            # Compute simple accuracy
            acc = sum(1 for t, p in zip(true_names, preds) if t == p) / len(true_names)
            results[p_name][str(lvl)] = {'accuracy': acc}
            print(f"Perturbation={p_name} lvl={lvl} -> Accuracy={acc*100:.2f}%")
            # Save incremental results so we keep progress on long runs
            out_path = Path('models/robustness_results.json')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)

    out_path = Path('models/robustness_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Robustness tests for webcam face recognizer")
    p.add_argument('--test_dir', default='data/test', help='Path to test dataset (structured by subject)')
    p.add_argument('--max_images', type=int, default=0, help='If >0, limit the number of test images (speeds up runs)')
    p.add_argument('--perturb_raw', action='store_true', help='Apply perturbations to raw images before preprocessing')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_tests(args)
