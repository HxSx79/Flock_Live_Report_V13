import cv2
from typing import Dict, List
from .flock_report import FlockReport

class LineDetector:
    def __init__(self):
        self.line_position = 0.5  # Middle of the frame
        self.previous_positions = {}
        self.flock_report = FlockReport()
        self.counted_ids = set()
        self.debug = True  # Enable debug printing

    def process_detections(self, frame: cv2.Mat, detections: List[Dict]) -> cv2.Mat:
        if frame is None:
            return frame

        height, width = frame.shape[:2]
        line_x = int(width * self.line_position)
        
        # Draw counting line
        cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 255), 2)
        
        for detection in detections:
            track_id = detection['track_id']
            class_name = detection['class_name']
            x1, y1, x2, y2 = detection['box']
            center_x = (x1 + x2) / 2
            
            # Draw center point for debugging
            cv2.circle(frame, (int(center_x), int((y1 + y2) / 2)), 4, (255, 0, 0), -1)
            
            if track_id in self.previous_positions:
                prev_x = self.previous_positions[track_id]
                
                # Check if crossed the line from either direction
                crossed_right = prev_x < line_x and center_x >= line_x
                crossed_left = prev_x > line_x and center_x <= line_x
                
                if (crossed_right or crossed_left) and track_id not in self.counted_ids:
                    if self.debug:
                        print(f"Line crossed by {class_name} (ID: {track_id})")
                        print(f"Previous X: {prev_x}, Current X: {center_x}, Line X: {line_x}")
                    
                    self.counted_ids.add(track_id)
                    self.flock_report.record_crossing(class_name)
                    
                    # Draw crossing indicator
                    direction = "→" if crossed_right else "←"
                    cv2.putText(frame, f"Crossed {direction}", 
                              (int(center_x), int((y1 + y2) / 2) - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            self.previous_positions[track_id] = center_x
        
        return frame

    def reset(self):
        """Reset tracking state"""
        self.previous_positions.clear()
        self.counted_ids.clear()