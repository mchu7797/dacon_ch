import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch import nn, optim
from sklearn.metrics import log_loss, accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Tuple, Dict

from .config import Config
from .car_dataset import CarDataset
from .model import ImprovedModel


def set_seed(seed: int):
    """완전한 시드 고정"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_transforms(config: Config) -> Tuple[transforms.Compose, transforms.Compose]:
    """개선된 데이터 증강"""
    train_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size + 32, config.image_size + 32)),
            transforms.RandomCrop(config.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


class EarlyStopping:
    """Early stopping 구현"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """한 에폭 훈련"""
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(train_loader)


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    """검증"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    logloss = log_loss(all_labels, all_probs, labels=list(range(num_classes)))

    return {"loss": avg_loss, "accuracy": accuracy, "log_loss": logloss}


def main():
    config = Config()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform, val_transform = get_transforms(config)

    # 임시 데이터셋으로 클래스 정보 얻기
    temp_dataset = CarDataset(config.train_root, transform=None)
    num_classes = len(temp_dataset.classes)
    targets = [label for _, label in temp_dataset.samples]
    print(f"Total samples: {len(temp_dataset)}, Classes: {num_classes}")

    # K-Fold 교차 검증
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(range(len(temp_dataset)), targets)
    ):
        print(f"\n{'=' * 50}")
        print(f"Fold {fold + 1}/5")
        print(f"{'=' * 50}")

        # 수정: 각 fold마다 새로운 데이터셋 생성
        train_dataset_full = CarDataset(config.train_root, transform=train_transform)
        val_dataset_full = CarDataset(config.train_root, transform=val_transform)

        train_dataset = Subset(train_dataset_full, train_idx)
        val_dataset = Subset(val_dataset_full, val_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

        # 모델 초기화
        model = ImprovedModel(num_classes=num_classes, model_name="efficientnet_b3").to(
            device
        )

        criterion = nn.CrossEntropyLoss(
            label_smoothing=0.1
        )  # 수정: label smoothing 추가
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        early_stopping = EarlyStopping(
            patience=config.patience, min_delta=config.min_delta
        )

        best_log_loss = float("inf")

        # 훈련 루프
        for epoch in range(1, config.num_epochs + 1):
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            val_metrics = validate_epoch(
                model, val_loader, criterion, device, num_classes
            )

            scheduler.step()

            print(f"Epoch {epoch}/{config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.2f}%")
            print(f"Val Log Loss: {val_metrics['log_loss']:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            # 모델 저장
            if val_metrics["log_loss"] < best_log_loss:
                best_log_loss = val_metrics["log_loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_log_loss": best_log_loss,
                        "config": config,
                        "class_names": temp_dataset.classes,  # 추가: 클래스명 저장
                    },
                    f"best_model_fold_{fold}.pth",
                )
                print(f"Best model saved! Log Loss: {best_log_loss:.4f}")

            # Early stopping
            if early_stopping(val_metrics["log_loss"]):
                print(f"Early stopping at epoch {epoch}")
                break

        fold_results.append(best_log_loss)
        print(f"Fold {fold + 1} Best Log Loss: {best_log_loss:.4f}")

    # 최종 결과 및 최고 모델 선택
    best_fold = np.argmin(fold_results)
    print(f"\n{'=' * 50}")
    print("Final Results:")
    print(f"Mean Log Loss: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"Best Fold: {best_fold + 1} (Log Loss: {fold_results[best_fold]:.4f})")
    print(f"Fold Results: {fold_results}")
    print(f"{'=' * 50}")

    # 최고 모델을 기본 이름으로 복사
    import shutil

    shutil.copy(f"best_model_fold_{best_fold}.pth", "best_model.pth")
    print("Best model copied as 'best_model.pth'")


if __name__ == "__main__":
    main()
