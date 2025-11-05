# Staff Name Tag Identification
This project implements a pipeline to detect and identify person wearing a name tag and determine them as staff members in a video. It combines YOLOv8 for person detection and pose estimated with ResNet18 model for staff classification.

## Project Overview
Step 1: Extract Images from Video for dataset (extract_image.py)

Step 2: Use dataset to train ResNet18 model (train_model.py)

Step 3: Detect person and identify whether they are staff (detect_staff.py)
