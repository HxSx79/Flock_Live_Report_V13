import cv2
import numpy as np
from typing import Optional, Tuple
from werkzeug.datastructures import FileStorage

class VideoStream:
    def __init__(self):
        self.cap = None
        self.test_video = None
        self.frame_interval = 1.0 / 25  # 25 FPS
        self.target_width = 1020
        self.target_height = 600

    def set_test_video(self, video_file: FileStorage) -> None:
        try:
            temp_path = "/tmp/test_video.mp4"
            video_file.save(temp_path)
            
            if self.test_video is not None:
                self.test_video.release()
            
            self.test_video = cv2.VideoCapture(temp_path)
            if not self.test_video.isOpened():
                raise ValueError("Failed to open video file")
                
            print("Test video loaded successfully")
            
        except Exception as e:
            print(f"Error setting test video: {e}")
            raise

    def maintain_aspect_ratio(self, frame):
        if frame is None:
            return None
            
        height, width = frame.shape[:2]
        aspect = width / height
        target_aspect = self.target_width / self.target_height
        
        if aspect > target_aspect:
            new_width = self.target_width
            new_height = int(self.target_width / aspect)
        else:
            new_height = self.target_height
            new_width = int(self.target_height * aspect)
            
        # Resize frame
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create canvas
        canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        # Center the frame
        y_offset = (self.target_height - new_height) // 2
        x_offset = (self.target_width - new_width) // 2
        
        # Copy the frame onto the canvas
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = frame
        
        return canvas

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.test_video is not None:
            ret, frame = self.test_video.read()
            if not ret:
                self.test_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.test_video.read()
            
            if ret:
                # Process frame before aspect ratio maintenance
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.maintain_aspect_ratio(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return ret, frame
        
        if self.cap is None:
            camera_id="/dev/video0"
            self.camera_id = camera_id
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 25)
        
        ret, frame = self.cap.read()
        if ret:
            frame = self.maintain_aspect_ratio(frame)
        return ret, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.test_video is not None:
            self.test_video.release()

    def generate_frames(self, detector):
        while True:
            success, frame = self.read_frame()
            if not success:
                break

            if frame is not None and detector is not None:
                frame = detector.process_frame(frame, skip_resize=True)

            if frame is not None:
                try:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error encoding frame: {e}")
                    continue