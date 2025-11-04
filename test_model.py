import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_folder = "person_images"  # Folder containing your test images

# --- Load trained ResNet18 ---
model_staff = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model_staff.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model_staff.fc.in_features, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(256, 1)  # single output for binary classification
)
model_staff.load_state_dict(torch.load("staff_resnet18.pth", map_location=device))
model_staff = model_staff.to(device)
model_staff.eval()

# --- Transform for ResNet18 input ---
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

# --- Helper function ---
def predict_tag(image_path):
    """Predict whether the person has a name tag (STAFF) or not."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Could not read {image_path}")
        return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = staff_transforms(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model_staff(input_tensor)
        prob = torch.sigmoid(logit).item()
    label = "STAFF" if prob > 0.5 else "NON-STAFF"
    return label, prob

# --- Run predictions on all images ---
results = []
for filename in sorted(os.listdir(image_folder)):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    img_path = os.path.join(image_folder, filename)
    label, prob = predict_tag(img_path)
    if label is None:
        continue
    print(f"{filename:30s} → {label} ({prob:.2f})")
    results.append({"filename": filename, "label": label, "probability": prob})

# --- Save results to CSV ---
df = pd.DataFrame(results)
print(df)

print("\n✅ Testing complete!")
