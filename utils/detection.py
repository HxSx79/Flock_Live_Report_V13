import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List
from .line_detector import LineDetector
from .production import ProductionTracker

class ObjectDetector:
    def __init__(self):
        self.model = YOLO('best.pt')
        self.line_detector = LineDetector()
        self.production_tracker = ProductionTracker()
        self.confidence_threshold = 0.95
        self.names = self.model.names
        self.debug = True

    def process_frame(self, frame: cv2.Mat, skip_resize: bool = False) -> cv2.Mat:
        if frame is None:
            return frame
            
        # Store original frame dimensions
        orig_height, orig_width = frame.shape[:2]
        
        # Process with YOLO
        results = self.model(frame, stream=True)
        detections = []
        
        # Process each detection
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy()
                else:
                    track_ids = np.arange(len(boxes))

                for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                    if conf < self.confidence_threshold:
                        continue
                        
                    class_name = self.names[int(class_id)]
                    x1, y1, x2, y2 = map(int, box)
                    
                    detection = {
                        'class_name': class_name,
                        'track_id': int(track_id),
                        'box': [x1, y1, x2, y2],
                        'confidence': float(conf)
                    }
                    detections.append(detection)
                    
                    # Draw detection box
                    color = (0, 255, 0) if class_name.endswith('_OK') else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with track ID and confidence
                    label = f'{int(track_id)} - {class_name} ({conf:.2f})'
                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    y1_label = max(y1, label_size[1])
                    
                    # Draw label background
                    cv2.rectangle(frame, 
                                (x1, y1_label - label_size[1] - baseline),
                                (x1 + label_size[0], y1_label + baseline),
                                color, 
                                cv2.FILLED)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1_label),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Process line crossings
        frame = self.line_detector.process_detections(frame, detections)
        
        return frame