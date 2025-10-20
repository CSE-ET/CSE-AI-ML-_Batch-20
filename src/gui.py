import cv2
import time
from utils.logger import setup_logger

logger = setup_logger()

class GUI:
    def __init__(self, recognizer, detector, preprocessor, tau):
        self.recognizer = recognizer
        self.detector = detector
        self.preprocessor = preprocessor
        self.tau = tau
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2

    def run(self, webcam_id=0):
        cap = cv2.VideoCapture(webcam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)
        
        fps_start = time.time()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            faces = self.detector.detect_faces(frame)
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                preprocessed = self.preprocessor.preprocess_face(face_roi)
                
                # ✅ FIX: Pass self.tau as second arg
                identity, conf = self.recognizer.recognize_face(preprocessed, self.tau)
                
                # Draw bounding box and label
                color = (0, 255, 0) if identity != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, self.thickness)
                cv2.putText(frame, f"{identity}: {conf:.2f}", (x, y-10),
                            self.font, self.font_scale, color, self.thickness)
            
            # FPS display
            if frame_count % 30 == 0:  # Update every 30 frames
                fps = frame_count / (time.time() - fps_start)
                logger.info(f"FPS: {fps:.2f}")
                frame_count = 0
                fps_start = time.time()
            
            cv2.imshow('Real-Time Facial Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()