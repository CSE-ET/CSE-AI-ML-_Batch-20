import argparse
import os
from pathlib import Path
import json
import shutil

import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

import cv2

from src.training import Trainer
from src.recognition import Recognizer


def save_lfw_to_dirs(lfw, target_root, min_faces=10, sample_limit=0):
    """Save LFW images into target_root/{person}/*.jpg for persons with >= min_faces.
    If sample_limit>0, limit total images saved to sample_limit (random).
    Returns list of saved file paths and corresponding labels.
    """
    target_root = Path(target_root)
    if target_root.exists():
        shutil.rmtree(target_root)
    (target_root / 'all').mkdir(parents=True, exist_ok=True)

    images, labels = lfw.images, lfw.target
    names = lfw.target_names

    # Build per-person lists
    per_person = {}
    for img, lbl in zip(images, labels):
        name = names[lbl].lower().replace(' ', '_')
        per_person.setdefault(name, []).append(img)

    # Filter
    selected = {n: imgs for n, imgs in per_person.items() if len(imgs) >= min_faces}
    all_saved = []
    all_labels = []
    for name, imgs in selected.items():
        dirp = target_root / name
        dirp.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(imgs):
            save_path = dirp / f"{name}_{i}.jpg"
            # lfw images are grayscale float arrays 0..255
            arr = img.astype('uint8')
            cv2.imwrite(str(save_path), arr)
            all_saved.append(str(save_path))
            all_labels.append(name)

    # Optionally sample down
    if sample_limit and sample_limit > 0 and len(all_saved) > sample_limit:
        idx = np.random.choice(len(all_saved), size=sample_limit, replace=False)
        sampled_paths = [all_saved[i] for i in idx]
        sampled_labels = [all_labels[i] for i in idx]
        # Create a small sampled tree under target_root/_sample/
        samp_root = target_root / '_sample'
        if samp_root.exists():
            shutil.rmtree(samp_root)
        for p, lbl in zip(sampled_paths, sampled_labels):
            dst_dir = samp_root / lbl
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / Path(p).name
            shutil.copy(p, dst)
        return samp_root

    return target_root


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min_faces', type=int, default=10)
    p.add_argument('--test_size', type=float, default=0.3)
    p.add_argument('--sample', type=int, default=0, help='If >0, sample this many images for a smoke test')
    p.add_argument('--work_dir', default='data/lfw')
    p.add_argument('--force_download', action='store_true')
    args = p.parse_args()

    print('Fetching LFW (this may download ~200MB)')
    lfw = fetch_lfw_people(color=False, resize=1.0, min_faces_per_person=0, download_if_missing=True)

    print('Saving images filtered by min_faces=%d' % args.min_faces)
    root = save_lfw_to_dirs(lfw, args.work_dir, min_faces=args.min_faces, sample_limit=args.sample)

    # If sampling created _sample, use that
    if (Path(root) / '_sample').exists():
        data_root = Path(root) / '_sample'
    else:
        data_root = Path(root)

    # Build train/test split by person directories
    persons = [d for d in sorted(data_root.iterdir()) if d.is_dir()]
    train_dir = Path('data/lfw_train')
    test_dir = Path('data/lfw_test')
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)

    for person in persons:
        imgs = list(person.glob('*.jpg'))
        if len(imgs) < 1:
            continue
        if len(imgs) < 2:
            # Not enough images for a test split; put into training set only
            for t in imgs:
                dst = train_dir / person.name
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(t, dst / Path(t).name)
            continue
        train_imgs, test_imgs = train_test_split([str(x) for x in imgs], test_size=args.test_size, random_state=42)
        for t in train_imgs:
            dst = train_dir / person.name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(t, dst / Path(t).name)
        for t in test_imgs:
            dst = test_dir / person.name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(t, dst / Path(t).name)

    print('Train dir:', train_dir, 'Test dir:', test_dir)

    # Train LBPH on train_dir
    trainer = Trainer()
    print('Loading training dataset...')
    # load_dataset will preprocess and save label_map
    trainer.load_dataset(str(train_dir))
    trainer.train()

    # Evaluate
    print('Loading test dataset for evaluation...')
    test_faces, test_labels = trainer.load_dataset(str(test_dir))
    recognizer = Recognizer(trainer)
    tau = recognizer.compute_adaptive_threshold(test_faces[:min(10, len(test_faces))])
    results = trainer.evaluate(test_faces, test_labels, recognizer, tau=tau)

    out = {'params': vars(args), 'results': results}
    Path('models').mkdir(parents=True, exist_ok=True)
    with open('models/lfw_eval.json', 'w') as f:
        json.dump(out, f, indent=2)

    print('Saved evaluation to models/lfw_eval.json')


if __name__ == '__main__':
    main()
