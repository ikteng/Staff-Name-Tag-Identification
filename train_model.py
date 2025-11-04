import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "dataset"   # should contain 0/ and 1/ subfolders
batch_size = 16
num_epochs = 50
model_path = "staff_resnet18.pth"
patience = 5

# --- Transforms ---
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.RandomErasing(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Dataset ---
full_dataset = datasets.ImageFolder(dataset_path)
indices = list(range(len(full_dataset)))
labels = [full_dataset.targets[i] for i in indices]

# --- Train / Validation split ---
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, stratify=labels, random_state=42
)

train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"✅ Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# --- Model: ResNet18 ---
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Unfreeze backbone
for param in model.parameters():
    param.requires_grad = True
print("🔓 Entire ResNet18 backbone unfrozen for full fine-tuning")

# Classifier for binary classification
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(256, 1)
)

model = model.to(device)

# --- Weighted loss for class imbalance ---
pos_count = sum([1 for i in train_idx if labels[i] == 1])
neg_count = sum([1 for i in train_idx if labels[i] == 0])
pos_weight = torch.tensor([neg_count / pos_count]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# --- Optimizer ---
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

# --- Training Loop ---
best_val_acc = 0.0
early_stop_counter = 0

for epoch in range(num_epochs):
    print(f"\n📘 Epoch {epoch+1}/{num_epochs}")

    # --- Train ---
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels_batch in tqdm(train_loader, desc="Training"):
        images, labels_batch = images.to(device), labels_batch.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels_batch).sum().item()
        total += labels_batch.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # --- Validate ---
    model.eval()
    val_correct = 0
    val_total = 0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for images, labels_batch in tqdm(val_loader, desc="Validating"):
            images, labels_batch = images.to(device), labels_batch.float().unsqueeze(1).to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (preds == labels_batch).sum().item()
            val_total += labels_batch.size(0)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels_batch.cpu().numpy())

    val_acc = val_correct / val_total
    val_f1 = f1_score(val_labels, val_preds)

    print(f"📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # --- Save best model ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved best model at epoch {epoch+1} (Val Acc: {val_acc:.4f})")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"⚠️ No improvement for {early_stop_counter} epoch(s)")
        if early_stop_counter >= patience:
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
            break
        
print(f"\n🏆 Training complete. Best validation accuracy: {best_val_acc:.4f}")

# --- Load the best model ---
model.load_state_dict(torch.load(model_path))
model.eval()

# --- Test on full dataset ---
test_dataset = full_dataset
test_dataset.transform = val_transforms
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels_batch in tqdm(test_loader, desc="Testing"):
        images, labels_batch = images.to(device), labels_batch.float().unsqueeze(1).to(device)
        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

# --- Metrics ---
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=['No Tag', 'Tag'])

print(f"✅ Test Accuracy: {acc:.4f}")
print(f"✅ Test F1 Score: {f1:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", report)
