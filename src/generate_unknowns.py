import shutil
from pathlib import Path

base_dir = Path("data/train")
unknown_base = base_dir / "unknown"
flat_unknown = base_dir / "unknown_flat"  # Temp
flat_unknown.mkdir(exist_ok=True)

all_imgs = []
for subdir in unknown_base.iterdir():
    if subdir.is_dir():
        imgs = list(subdir.glob("*.*"))  # All images (jpg/png/etc.)
        all_imgs.extend(imgs)
        print(f"Found {len(imgs)} images in {subdir.name}")

print(f"Total to flatten: {len(all_imgs)}")

saved = 0
for i, img_path in enumerate(all_imgs):
    try:
        # Copy & rename
        dest = flat_unknown / f"unk{i+1:03d}.jpg"
        shutil.copy2(img_path, dest)
        saved += 1
        print(f"Copied {img_path.name} → {dest.name}")
    except Exception as e:
        print(f" Skip {img_path.name}: {e}")

if saved > 0:
    # Replace unknown/ with flat
    shutil.rmtree(unknown_base)
    flat_unknown.rename(unknown_base)
    print(f"\nFlattened {saved} unknowns into data/train/unknown/ (flat structure)")
else:
    print("No images—check subfolders.")

print("Retrain: python -m src.main --mode train")