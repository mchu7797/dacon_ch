import os
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from tqdm import tqdm
from torch.amp import GradScaler, autocast

from config import Config, get_config
from models import get_models
from datasets import CarDataset
from utils import fix_random_seed, show_dataset_info
from transforms import get_train_transform, get_val_transform


def train_model(
    config: Config,
    model: torch.nn.Module,
    class_names: list,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device_name: str,
    fold: int = 0,
):
    best_logloss = float("inf")
    patience_counts = 0

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(
            train_loader, desc=f"[Epoch {epoch + 1}/{config.epochs}] Training"
        ):
            images, labels = images.to(device_name), labels.to(device_name)

            optimizer.zero_grad()
            
            with autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        average_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(
                val_loader, desc=f"[Epoch {epoch + 1}/{config.epochs}] Validation"
            ):
                images, labels = images.to(device_name), labels.to(device_name)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                probs = torch.softmax(outputs, dim=1)
                probs = probs / probs.sum(dim=1, keepdim=True)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        average_val_loss = val_loss / len(val_loader)
        accuracy = correct / total * 100
        val_logloss = log_loss(
            all_labels, all_probs, labels=list(range(len(class_names)))
        )

        print(
            f"Epoch [{epoch + 1}/{config.epochs}] - "
            f"Train Loss: {average_train_loss:.4f}, "
            f"Val Loss: {average_val_loss:.4f}, "
            f"Accuracy: {accuracy:.2f}%, "
            f"Log Loss: {val_logloss:.4f}"
        )

        improvement = best_logloss - val_logloss

        if improvement > config.early_stopping_minimum_delta:
            best_logloss = val_logloss
            patience_counts = 0

            os.makedirs(config.model_directory, exist_ok=True)

            torch.save(
                model.state_dict(),
                f"{config.model_directory}/best_{model.__class__.__name__}_fold{fold}.pth",
            )
            print(f"Best model for fold {fold} saved with Log Loss: {best_logloss:.4f}")
        elif config.early_stopping_enabled:
            patience_counts += 1
            print(
                f"No improvement. Patience [{patience_counts}/{config.early_stopping_patience}]"
            )

            if patience_counts >= config.early_stopping_patience:
                print("Early stopping triggered. stopping")
                return


def train():
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    fix_random_seed(config.seed)

    # 데이터셋은 transform 없이 한 번만 로드하여 인덱싱에 사용
    full_dataset = CarDataset(config.train_directory, transform=None)
    labels = [label for _, label in full_dataset.data]
    class_names = full_dataset.classes

    show_dataset_info(full_dataset)

    # Stratified K-Fold 설정
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)

    # K-Fold 루프 시작
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(full_dataset)), labels)):
        print(f"\n===== Starting Fold {fold} =====\n")

        # 각 Fold에 맞는 데이터셋과 DataLoader 생성
        train_dataset = Subset(
            CarDataset(
                config.train_directory,
                transform=get_train_transform(config.image_size, config.mean, config.std),
            ),
            indices=train_idx,
        )
        val_dataset = Subset(
            CarDataset(
                config.train_directory,
                transform=get_val_transform(config.image_size, config.mean, config.std),
            ),
            indices=val_idx,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        # 각 Fold마다 새로운 모델을 초기화하여 학습
        models_for_fold = get_models(num_classes=len(class_names))

        for model in models_for_fold:
            model.to(device)
            print(f"Training model: {model.__class__.__name__} for Fold {fold}")
            # train_model 함수에 fold 번호 전달
            train_model(config, model, class_names, train_loader, val_loader, device, fold)

if __name__ == "__main__":
    train()