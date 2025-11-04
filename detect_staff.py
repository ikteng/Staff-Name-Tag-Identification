import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# --- Load YOLO models ---
person_model = YOLO('yolov8m.pt')        # Person detection (COCO)
pose_model = YOLO('yolov8m-pose.pt')     # Pose estimation (keypoints)

# --- Video setup ---
video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Processing {frame_count} frames at {fps:.2f} FPS...")

# Output video setup
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("annotated_output.mp4", fourcc, fps, (width, height))

# Folder to save frames with staff
staff_frame_dir = "staff_frames"
os.makedirs(staff_frame_dir, exist_ok=True)

# --- Device & Staff Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained ResNet18
model_staff = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model_staff.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model_staff.fc.in_features, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(256, 1)  # single output for binary classification
)
# Load trained weights from checkpoint
model_staff.load_state_dict(torch.load("staff_resnet18.pth", map_location=device))
model_staff = model_staff.to(device)
model_staff.eval()

# Transform for ResNet18 input
staff_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def has_tag(crop):
    """Detects if a crop likely belongs to staff using the trained ResNet18 model."""
    if crop is None or crop.size == 0:
        return False
    input_tensor = staff_transforms(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        staff_prob = torch.sigmoid(model_staff(input_tensor)).item()
    return staff_prob > 0.5

def annotate_frame(frame, box, label="STAFF"):
    """Draws a rectangle, label, and center coordinates on the frame."""
    x1, y1, x2, y2 = box
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{label} ({x_center},{y_center})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.circle(frame, (x_center, y_center), 4, (0, 255, 0), -1)
    return x_center, y_center

def draw_pose(frame, keypoints, conf_threshold=0.3):
    # Define keypoint pairs to connect with lines (basic skeleton)
    skeleton_pairs = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 6),
        (11, 12),
        (5, 11), (6, 12)
    ]
    keypoints = np.array(keypoints).reshape(-1, 3)

    # Draw keypoints
    for x, y, c in keypoints:
        if c > conf_threshold:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
    
    # Draw skeleton connections
    for i1, i2 in skeleton_pairs:
        if keypoints[i1][2] > conf_threshold and keypoints[i2][2] > conf_threshold:
            cv2.line(frame, tuple(keypoints[i1][:2].astype(int)), tuple(keypoints[i2][:2].astype(int)), (255, 0, 0), 2)

def is_valid_pose(keypoints, conf_threshold=0.5):
    keypoints = np.array(keypoints).reshape(-1, 3)
    conf = keypoints[:, 2]

    # Must have confident shoulders
    if conf[5] < conf_threshold or conf[6] < conf_threshold:
        return False
    
    # Must have at least one confident hip
    if conf[11] < conf_threshold and conf[12] < conf_threshold:
        return False
    
    # Check orientation (head should be above hips)
    head_y = keypoints[0][1] if conf[0] > conf_threshold else None
    hips = [kp[1] for kp in keypoints[[11, 12]] if kp[2] > conf_threshold]
    hips_y = np.mean(hips) if hips else None
    if head_y is not None and hips_y is not None and head_y > hips_y:
        return False
    
    # Reject if shoulders are too close or far apart (bad detection)
    shoulder_dist = abs(keypoints[5][0] - keypoints[6][0])
    if shoulder_dist < 20 or shoulder_dist > 300:
        return False
    
    # Require enough visible keypoints
    if (conf > conf_threshold).sum() < 5:
        return False
    return True

# --- Frame processing ---
output = [] # Will store frame number + staff coordinates

for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
    ret, frame = cap.read() # Read one frame from the video
    if not ret:
        break # Stop if no more frame

    # --- Split into top and bottom halves ---
    top_half = frame[:height // 2, :]
    bottom_half = frame[height // 2:, :]
    bottom_half_flipped = cv2.flip(bottom_half, 0) # flip vertically
    combined = np.vstack((top_half, bottom_half_flipped))

    # --- Run person detection using YOLOv8 ---
    detections = person_model(combined, verbose=False)
    frame_staff_coords = []
    staff_detected_this_frame = False

    for r in detections:
        for box in r.boxes:
            # Keep only person with confidence > 0.5
            if int(box.cls) != 0 or float(box.conf) < 0.5:
                continue

            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Crop out the detected person from the frame
            crop = combined[y1:y2, x1:x2]
            if crop.size == 0:
                continue # Skip empty or invalid crops

            # Pose estimation
            pose_results = pose_model.predict(crop, verbose=False)
            valid_pose = False
            keypoints_for_draw = None

            # Extract keypoints from pose result
            for pr in pose_results:
                if pr.keypoints is not None and len(pr.keypoints.xy) > 0:
                    keypoints_xy = pr.keypoints.xy[0].cpu().numpy() # (x, y) keypoint positions
                    keypoints_conf = pr.keypoints.conf[0].cpu().numpy() # Confidence values for each keypoint
                    keypoints_combined = np.hstack([keypoints_xy, keypoints_conf.reshape(-1,1)]) # Combine into one array
                    
                    # Check if pose is valid based on keypoint confidence and structure
                    if is_valid_pose(keypoints_combined):
                        valid_pose = True
                        keypoints_for_draw = keypoints_combined
                        break
            if not valid_pose:
                continue
            
            # Draw skeleton pose on frame
            draw_pose(combined, keypoints_for_draw + np.array([x1, y1, 0]))

            # Extract Upperbody crop (based on keypoints)
            idx_map = {"left_shoulder":5,"right_shoulder":6,"left_hip":11,"right_hip":12}
            points_list = [keypoints_for_draw[idx][:2] for idx in idx_map.values() if keypoints_for_draw[idx][2] > 0.3]
            if len(points_list) < 2:
                continue # Skip if too few keypoints detected
            points = np.array(points_list)

             # Compute bounding box of upper body area
            x_min = int(np.min(points[:,0])) + x1
            x_max = int(np.max(points[:,0])) + x1
            y_min = int(np.min(points[:,1])) + y1
            y_max = int(np.max(points[:,1])) + y1

            # Extract upper body region
            upperbody_crop = combined[y_min:y_max, x_min:x_max]
            if upperbody_crop.size == 0 or upperbody_crop.shape[1]<30 or upperbody_crop.shape[0]<30:
                continue # Skip if crop too small or invalid

            # Classify as staff/non staff using ResNet18
            if has_tag(upperbody_crop):
                staff_coords = annotate_frame(combined, (x1, y1, x2, y2))
                frame_staff_coords.append(staff_coords)
                staff_detected_this_frame = True

    # Restore bottom half orientation & write frame
    restored_bottom = cv2.flip(combined[height//2:,:], 0)
    frame_output = np.vstack((combined[:height//2,:], restored_bottom))
    out.write(frame_output)

    # Save frame if staff detected
    if staff_detected_this_frame:
        save_path = os.path.join(staff_frame_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(save_path, frame_output)

    # Record staff position for this frame
    for coords in frame_staff_coords:
        output.append({"frame": frame_idx, "x": coords[0], "y": coords[1]})

# Cleanup
cap.release()
out.release()

# Save staff coordinates as CSV
df = pd.DataFrame(output)
df.to_csv("staff_positions.csv", index=False)

print("\n✅ Detection + Pose + Upperbody Tag completed!")
print(f"📄 Results saved to: staff_positions.csv")
print(f"🎥 Annotated video saved to: annotated_output.mp4")
print(f"📸 Frames with staff saved to: {staff_frame_dir}/")
