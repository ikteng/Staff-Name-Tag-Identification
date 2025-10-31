import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8m.pt')
cap = cv2.VideoCapture("sample.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]
    h2, w2 = height // 2, width // 2
    
    annotated_frame = frame.copy()
    
    # Process each quadrant with custom strategies
    quadrants = [
        # (x1, y1, x2, y2, is_left, is_bottom)
        (0, 0, w2, h2, True, False),    # Top-left
        (w2, 0, width, h2, False, False), # Top-right
        (0, h2, w2, height, True, True),  # Bottom-left  
        (w2, h2, width, height, False, True) # Bottom-right
    ]
    
    colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    
    for i, (x1, y1, x2, y2, is_left, is_bottom) in enumerate(quadrants):
        quadrant = frame[y1:y2, x1:x2]
        
        # Special handling for left quadrants (more distorted)
        if is_left:
            # Try multiple approaches for left quadrants
            conf_threshold = 0.3  # Lower confidence for distorted images
            imgsz = 640  # Larger input size
        else:
            conf_threshold = 0.4
            imgsz = 480
        
        if is_bottom:
            quadrant = cv2.rotate(quadrant, cv2.ROTATE_180)
        
        results = model(quadrant, conf=conf_threshold, imgsz=imgsz)
        
        for result in results:
            for box in result.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                if class_id == 0:  # Person
                    # Adjust coordinates
                    if is_bottom:
                        q_height, q_width = quadrant.shape[:2]
                        bx1, by1, bx2, by2 = q_width-bx2, q_height-by2, q_width-bx1, q_height-by1
                    
                    # Map back to original frame
                    bx1 += x1
                    bx2 += x1
                    by1 += y1
                    by2 += y1
                    
                    color = colors[i]
                    cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 3)
                    label = f"Person {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (bx1, by1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw grid
    cv2.line(annotated_frame, (0, h2), (width, h2), (255, 255, 255), 2)
    cv2.line(annotated_frame, (w2, 0), (w2, height), (255, 255, 255), 2)
    cv2.imshow('Adaptive Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()