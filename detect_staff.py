import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# --- Load YOLO model ---
model = YOLO('yolov8n.pt')  # Pretrained COCO model (person class)

# --- Video setup ---
video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[INFO] Processing {frame_count} frames at {fps:.2f} FPS...")

# --- Output setup ---
output = []
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("annotated_output_detect.mp4", fourcc, fps, (width, height))

# --- Tag detection helper ---
def has_tag(person_crop):
    """Detects if a person crop likely has a white rectangular tag on the chest."""
    if person_crop.size == 0:
        return False
    hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))  # white color range
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = w / float(h + 1e-5)
        if 1.0 < aspect < 3.0 and 150 < area < 2000:
            return True
    return False

# --- Frame-by-frame processing ---
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    staff_found = False
    staff_coords = None

    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]

                # Focus on upper chest region (30–60% of height)
                chest_crop = crop[int(0.3 * crop.shape[0]):int(0.6 * crop.shape[0]), :]

                if has_tag(chest_crop):
                    staff_found = True
                    x_center = int((x1 + x2) / 2)
                    y_center = int((y1 + y2) / 2)
                    staff_coords = (x_center, y_center)

                    # Draw bounding box + label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "STAFF", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.circle(frame, staff_coords, 4, (0, 255, 0), -1)
                    break  # assume one staff per frame

    # Save frame to output video
    out.write(frame)

    # Save frame data
    output.append({
        "frame": frame_idx,
        "staff_present": staff_found,
        "x": staff_coords[0] if staff_coords else None,
        "y": staff_coords[1] if staff_coords else None
    })

    frame_idx += 1
    if frame_idx % 100 == 0:
        print(f"[INFO] Processed {frame_idx}/{frame_count} frames...")

# --- Cleanup ---
cap.release()
out.release()

# --- Save results ---
pd.DataFrame(output).to_csv("staff_positions.csv", index=False)
print("\n✅ Detection completed!")
print("📄 Results saved to: staff_positions.csv")
print("🎥 Annotated video saved to: annotated_output.mp4")
