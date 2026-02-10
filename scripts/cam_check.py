import cv2
import time
import sys

# Simple camera diagnostic: tries indices 0-4 and common backends (DSHOW/MSMF on Windows)
def try_open(index, backend=None):
    try:
        if backend is None:
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, backend)
    except Exception as e:
        print(f"Index {index} backend {backend}: open() raised {e}")
        return False
    if not cap.isOpened():
        return False
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return False
    h, w = frame.shape[:2]
    cap.release()
    print(f"Index {index} backend {backend}: OK (frame {w}x{h})")
    return True

if __name__ == '__main__':
    print("Camera diagnostic — will probe indices 0..4 with common backends.")
    success = False
    for i in range(5):
        if try_open(i):
            success = True
        if try_open(i, cv2.CAP_DSHOW):
            success = True
        if try_open(i, cv2.CAP_MSMF):
            success = True
    if not success:
        print("No usable camera found. Suggestions:")
        print(" - Ensure no other app is using the camera (close Zoom/Teams).")
        print(" - On Windows, check Camera privacy settings and allow apps access.")
        print(" - Try running Python as Administrator if permissions block access.")
        print(" - Try a different webcam index: run `python scripts/cam_check.py` and note working indices.")
        sys.exit(2)
    else:
        print("At least one camera opened successfully.")
