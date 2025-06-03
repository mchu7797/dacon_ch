import os
import random
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, optim
import timm
import torchvision.transforms as transforms
from sklearn.metrics import log_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameter Setting
CFG = {
    "IMG_SIZE": 224,
    "BATCH_SIZE": 16,
    "EPOCHS": 20,
    "LEARNING_RATE": 1e-4,
    "SEED": 42,
}


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG["SEED"])


# Brand Dataset for hierarchical structure
class BrandDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Extract brand classes
        self.brands = sorted(os.listdir(root_dir))
        self.brand_to_idx = {brand: i for i, brand in enumerate(self.brands)}

        # Collect all images with brand labels
        for brand in self.brands:
            brand_folder = os.path.join(root_dir, brand)
            for model_folder in os.listdir(brand_folder):
                model_path = os.path.join(brand_folder, model_folder)
                if os.path.isdir(model_path):
                    for fname in os.listdir(model_path):
                        if fname.lower().endswith(".jpg"):
                            img_path = os.path.join(model_path, fname)
                            brand_label = self.brand_to_idx[brand]
                            self.samples.append((img_path, brand_label, fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, brand_label, fname = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, brand_label, fname


# ConvNeXt Brand Model - 수정된 버전
class BrandModel(nn.Module):
    def __init__(self, num_brands):
        super(BrandModel, self).__init__()
        self.model = timm.create_model(
            "timm/convnextv2_base.fcmae_ft_in22k_in1k",
            pretrained=True,
            num_classes=num_brands,
        )

    def forward(self, x):
        return self.model(x)


# Data transforms
train_transform = transforms.Compose(
    [
        transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load brand dataset
brand_root = "./car_type_train"
full_brand_dataset = BrandDataset(brand_root, transform=None)
print(f"총 브랜드 이미지 수: {len(full_brand_dataset)}")
print(f"브랜드 클래스 수: {len(full_brand_dataset.brands)}")
print(f"브랜드 목록: {full_brand_dataset.brands}")

# Extract targets for stratified split
targets = [sample[1] for sample in full_brand_dataset.samples]

# Stratified split
train_idx, val_idx = train_test_split(
    range(len(targets)), test_size=0.2, stratify=targets, random_state=42
)


# Create datasets with transforms
class BrandSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img_path, brand_label, fname = self.dataset.samples[original_idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, brand_label, fname


train_brand_dataset = BrandSubset(full_brand_dataset, train_idx, train_transform)
val_brand_dataset = BrandSubset(full_brand_dataset, val_idx, val_transform)

print(
    f"브랜드 train 이미지 수: {len(train_brand_dataset)}, valid 이미지 수: {len(val_brand_dataset)}"
)

# DataLoaders
train_brand_loader = DataLoader(
    train_brand_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=True
)
val_brand_loader = DataLoader(
    val_brand_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=False
)

# Initialize brand model
brand_model = BrandModel(num_brands=len(full_brand_dataset.brands)).to(device)
brand_criterion = nn.CrossEntropyLoss()
brand_optimizer = optim.Adam(brand_model.parameters(), lr=CFG["LEARNING_RATE"])

# Train brand model
print("\n=== 브랜드 모델 학습 시작 ===")
best_brand_logloss = float("inf")

for epoch in range(CFG["EPOCHS"]):
    # Train
    brand_model.train()
    train_loss = 0.0

    for images, labels, _ in tqdm(
        train_brand_loader, desc=f"[Epoch {epoch + 1}/{CFG['EPOCHS']}] Brand Training"
    ):
        images, labels = images.to(device), labels.to(device)
        brand_optimizer.zero_grad()
        outputs = brand_model(images)
        loss = brand_criterion(outputs, labels)
        loss.backward()
        brand_optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_brand_loader)

    # Validation
    brand_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(
            val_brand_loader,
            desc=f"[Epoch {epoch + 1}/{CFG['EPOCHS']}] Brand Validation",
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = brand_model(images)
            loss = brand_criterion(outputs, labels)
            val_loss += loss.item()

            # Accuracy
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # LogLoss
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_brand_loader)
    val_accuracy = 100 * correct / total
    val_logloss = log_loss(
        all_labels, all_probs, labels=list(range(len(full_brand_dataset.brands)))
    )

    print(
        f"Brand Train Loss: {avg_train_loss:.4f} || Valid Loss: {avg_val_loss:.4f} | Valid Accuracy: {val_accuracy:.4f}%"
    )

    # Best model 저장
    if val_logloss < best_brand_logloss:
        best_brand_logloss = val_logloss
        torch.save(brand_model.state_dict(), "brand_model.pth")
        print(
            f"📦 Best brand model saved at epoch {epoch + 1} (logloss: {val_logloss:.4f})"
        )

# Load best brand model for inference
brand_model.load_state_dict(torch.load("brand_model.pth", map_location=device))
brand_model.eval()

# Generate brand predictions for all images
print("\n=== 모든 이미지에 대한 브랜드 예측 생성 ===")

# Create dataset for all images (without train/val split)
all_dataset = BrandDataset(brand_root, transform=val_transform)
all_loader = DataLoader(all_dataset, batch_size=CFG["BATCH_SIZE"], shuffle=False)

brand_predictions = {}

with torch.no_grad():
    for images, _, fnames in tqdm(all_loader, desc="Generating brand predictions"):
        images = images.to(device)
        outputs = brand_model(images)

        # Store predictions with filename as key
        for output, fname in zip(outputs.cpu().numpy(), fnames):
            brand_predictions[fname] = output

# Save brand predictions
with open("brand_predictions.pkl", "wb") as f:
    pickle.dump(brand_predictions, f)

print(f"브랜드 예측 결과 저장 완료: {len(brand_predictions)}개 이미지")
print(f"브랜드 클래스 수: {len(full_brand_dataset.brands)}")
print("저장된 파일: brand_predictions.pkl, brand_model.pth")

# Save brand class mapping for later use
brand_info = {
    "brands": full_brand_dataset.brands,
    "brand_to_idx": full_brand_dataset.brand_to_idx,
}

with open("brand_info.pkl", "wb") as f:
    pickle.dump(brand_info, f)

print("브랜드 정보 저장 완료: brand_info.pkl")
