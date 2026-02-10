#!/usr/bin/env python3
"""Import images from filesystem (data/train, data/test) into SQLite DB used by the project.

Usage:
  python scripts/import_fs_to_db.py --train_dir data/train --test_dir data/test --db models/images.db --commit

Options:
  --train_dir   Path to train folder (default: data/train)
  --test_dir    Path to test folder (default: data/test)
  --db          DB path (default: models/images.db)
  --commit      If present, perform inserts; otherwise run as dry-run and report counts.
"""
import argparse
from pathlib import Path
import sys

# Ensure repo root is on sys.path so `utils` can be imported when the script is
# executed directly from `scripts/`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import db as db_helper


def iter_images(base_dir: Path):
    """Yield tuples (subject_name, file_path)"""
    if not base_dir.exists():
        return
    for subject_dir in sorted(base_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name
        for img_path in sorted(subject_dir.glob('*.jpg')):
            yield subject, img_path


def main(argv=None):
    p = argparse.ArgumentParser(description="Import images from filesystem into DB")
    p.add_argument('--train_dir', default='data/train')
    p.add_argument('--test_dir', default='data/test')
    p.add_argument('--db', default='models/images.db')
    p.add_argument('--commit', action='store_true', help='Perform DB inserts. Without this flag the script runs as dry-run')
    args = p.parse_args(argv)

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    db_path = args.db

    # Ensure DB exists
    db_helper.init_db(db_path)

    summary = {'train': {}, 'test': {}}
    # Count train images
    for subject, img_path in iter_images(train_dir):
        summary['train'].setdefault(subject, 0)
        summary['train'][subject] += 1

    for subject, img_path in iter_images(test_dir):
        summary['test'].setdefault(subject, 0)
        summary['test'][subject] += 1

    print('Dry-run summary:')
    print('Train subjects:')
    for s, cnt in summary['train'].items():
        print(f'  {s}: {cnt} images')
    print('Test subjects:')
    for s, cnt in summary['test'].items():
        print(f'  {s}: {cnt} images')

    if not args.commit:
        print('\nRun again with --commit to perform the import into', db_path)
        return 0

    # Commit: import images
    inserted = 0
    for subject, img_path in iter_images(train_dir):
        with open(img_path, 'rb') as f:
            data = f.read()
        try:
            db_helper.add_image(subject, data, is_train=True, filename=img_path.name, db_path=db_path)
            inserted += 1
        except Exception as e:
            print('Failed to insert', img_path, e)

    for subject, img_path in iter_images(test_dir):
        with open(img_path, 'rb') as f:
            data = f.read()
        try:
            db_helper.add_image(subject, data, is_train=False, filename=img_path.name, db_path=db_path)
            inserted += 1
        except Exception as e:
            print('Failed to insert', img_path, e)

    print(f'Inserted {inserted} images into {db_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
