import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO('yolov8m.pt')

# Input video
cap = cv2.VideoCapture("sample.mp4")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output writer (you can change 'mp4v' → 'XVID' if needed)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('annotated_output_test.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    h2, w2 = h // 2, w // 2
    
    annotated_frame = frame.copy()
    
    # Define quadrants
    quadrants = [
        (0, 0, w2, h2, True, False),       # Top-left
        (w2, 0, w, h2, False, False),      # Top-right
        (0, h2, w2, h, True, True),        # Bottom-left
        (w2, h2, w, h, False, True)        # Bottom-right
    ]
    
    colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    
    for i, (x1, y1, x2, y2, is_left, is_bottom) in enumerate(quadrants):
        quadrant = frame[y1:y2, x1:x2]
        
        # Different settings per quadrant
        conf_threshold = 0.3 if is_left else 0.4
        imgsz = 640 if is_left else 480
        
        if is_bottom:
            quadrant = cv2.rotate(quadrant, cv2.ROTATE_180)
        
        results = model(quadrant, conf=conf_threshold, imgsz=imgsz)
        
        for result in results:
            for box in result.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                
                if class_id == 0:  # Person
                    # Adjust for rotated quadrant
                    if is_bottom:
                        qh, qw = quadrant.shape[:2]
                        bx1, by1, bx2, by2 = qw-bx2, qh-by2, qw-bx1, qh-by1
                    
                    # Map back to original coordinates
                    bx1 += x1
                    bx2 += x1
                    by1 += y1
                    by2 += y1
                    
                    # Chest region (middle 1/3 of body)
                    person_h = by2 - by1
                    chest_h = person_h // 3
                    chest_x1, chest_y1 = bx1, by1 + chest_h
                    chest_x2, chest_y2 = bx2, by1 + 2 * chest_h
                    
                    color = colors[i]
                    
                    # Draw body box
                    cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 1)
                    # Draw chest box
                    cv2.rectangle(annotated_frame, (chest_x1, chest_y1), (chest_x2, chest_y2), color, 3)
                    cv2.putText(
                        annotated_frame,
                        f"Chest {confidence:.2f}",
                        (chest_x1, chest_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
    
    # Draw grid
    cv2.line(annotated_frame, (0, h2), (w, h2), (255, 255, 255), 2)
    cv2.line(annotated_frame, (w2, 0), (w2, h), (255, 255, 255), 2)

    # Write frame to output video
    out.write(annotated_frame)

    # # Optional: display live preview
    # cv2.imshow('Chest Detection', annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()
