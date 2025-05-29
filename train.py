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
import shutil
import gc
import psutil

from config import Config
from car_dataset import CarDataset
from model import Model


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


def check_disk_space() -> bool:
    """디스크 공간 확인"""
    try:
        disk_usage = psutil.disk_usage(".")
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 5:  # 5GB 미만이면 경고
            print(f"⚠️ Warning: Low disk space ({free_gb:.1f}GB remaining)")
            return False
        return True
    except Exception as e:
        print(f"❌ Failed to check disk space: {e}")
        return True


def get_transforms(
    config: Config,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """데이터 증강 (앙상블용으로 조정)"""
    train_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size + 32, config.image_size + 32)),
            transforms.RandomCrop(config.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # 큰 모델이므로 약간 감소
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05
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

    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
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
        try:
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

            # 메모리 정리 (매 50 배치마다)
            if batch_idx % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️ OOM at batch {batch_idx}, clearing cache and skipping...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e

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
        for batch_idx, (images, labels) in enumerate(
            tqdm(val_loader, desc="Validating")
        ):
            try:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # 메모리 정리
                del images, labels, outputs, probs, preds
                if batch_idx % 20 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ OOM during validation at batch {batch_idx}, skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    logloss = log_loss(all_labels, all_probs, labels=list(range(num_classes)))

    return {"loss": avg_loss, "accuracy": accuracy, "log_loss": logloss}


def train_single_model(
    model_config: Dict,
    fold: int,
    train_dataset: Subset,
    val_dataset: Subset,
    config: Config,
    num_classes: int,
    device: torch.device,
) -> str:
    """단일 모델 훈련"""

    print(f"\n🚀 Training {model_config['name']} - Fold {fold + 1}")

    # 디스크 공간 확인
    if not check_disk_space():
        print("⚠️ Warning: Low disk space detected")

    # 데이터 로더
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
    try:
        model = Model(
            num_classes=num_classes,
            model_name=model_config["model_name"],
            pretrained=model_config["pretrained"],
            dropout_rate=model_config["dropout_rate"],
            img_size=config.image_size,
        ).to(device)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ OOM during model initialization, reducing batch size...")
            config.batch_size = max(2, config.batch_size // 2)
            raise e
        else:
            raise e

    # 옵티마이저 및 스케줄러
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=1e-7)
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)

    best_log_loss = float("inf")
    model_save_path = f"ensemble_{model_config['name']}_fold_{fold}.pth"

    # 훈련 루프
    for epoch in range(1, config.num_epochs + 1):
        try:
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            val_metrics = validate_epoch(
                model, val_loader, criterion, device, num_classes
            )

            scheduler.step()

            print(
                f"  Epoch {epoch:2d}/{config.num_epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Val: {val_metrics['loss']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.2f}% | "
                f"LogLoss: {val_metrics['log_loss']:.4f}"
            )

            # 모델 저장
            if val_metrics["log_loss"] < best_log_loss:
                best_log_loss = val_metrics["log_loss"]

                try:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_log_loss": best_log_loss,
                            "config": config,
                            "model_config": model_config,
                        },
                        model_save_path,
                    )
                    print(f"    💾 Best model saved! LogLoss: {best_log_loss:.4f}")
                except Exception as e:
                    print(f"⚠️ Failed to save model: {e}")

            # Early stopping
            if early_stopping(val_metrics["log_loss"]):
                print(f"    ⏹️ Early stopping at epoch {epoch}")
                break

            # 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"❌ Error during epoch {epoch}: {e}")
            if "out of memory" in str(e):
                print("💡 Trying to continue with memory cleanup...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e

    print(
        f"✅ {model_config['name']} training completed! Best LogLoss: {best_log_loss:.4f}"
    )
    return model_save_path


def main():
    config = Config()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB"
        )

    print(f"📊 Ensemble models: {[m['name'] for m in config.ensemble_models]}")

    # 데이터 준비
    try:
        train_transform, val_transform = get_transforms(config)
        temp_dataset = CarDataset(config.train_root, transform=None)
        num_classes = len(temp_dataset.classes)
        targets = [label for _, label in temp_dataset.samples]
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    print(f"📁 Total samples: {len(temp_dataset)}, Classes: {num_classes}")

    # K-Fold 교차 검증으로 각 모델 훈련
    skf = StratifiedKFold(
        n_splits=config.n_folds, shuffle=True, random_state=config.seed
    )
    all_results = {}

    for model_config in config.ensemble_models:
        model_name = model_config["name"]
        print(f"\n{'=' * 60}")
        print(f"🎯 Training {model_name}")
        print(f"{'=' * 60}")

        fold_results = []
        model_paths = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(range(len(temp_dataset)), targets)
        ):
            try:
                # 데이터셋 생성
                train_dataset_full = CarDataset(
                    config.train_root, transform=train_transform
                )
                val_dataset_full = CarDataset(
                    config.train_root, transform=val_transform
                )

                train_dataset = Subset(train_dataset_full, train_idx)
                val_dataset = Subset(val_dataset_full, val_idx)

                # 모델 훈련
                model_path = train_single_model(
                    model_config,
                    fold,
                    train_dataset,
                    val_dataset,
                    config,
                    num_classes,
                    device,
                )

                # 결과 기록
                try:
                    checkpoint = torch.load(model_path, map_location="cpu")
                    fold_results.append(checkpoint["val_log_loss"])
                    model_paths.append(model_path)
                    del checkpoint
                except Exception as e:
                    print(f"⚠️ Failed to load checkpoint for fold {fold}: {e}")
                    continue

                # 메모리 정리
                del train_dataset, val_dataset, train_dataset_full, val_dataset_full
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"❌ Failed training fold {fold} for {model_name}: {e}")
                continue

        if fold_results:  # 성공한 fold가 있는 경우만
            # 모델별 결과 저장
            all_results[model_name] = {
                "fold_results": fold_results,
                "mean_score": np.mean(fold_results),
                "std_score": np.std(fold_results),
                "best_fold": np.argmin(fold_results),
                "model_paths": model_paths,
            }

            print(f"\n📈 {model_name} Results:")
            print(
                f"   Mean LogLoss: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}"
            )
            print(
                f"   Best Fold: {np.argmin(fold_results) + 1} (LogLoss: {min(fold_results):.4f})"
            )

    # 최종 결과 출력 및 최고 모델들 선택
    if all_results:
        print(f"\n{'=' * 60}")
        print("🏆 FINAL ENSEMBLE RESULTS")
        print(f"{'=' * 60}")

        selected_models = []
        for model_name, results in all_results.items():
            best_fold = results["best_fold"]
            best_path = results["model_paths"][best_fold]
            final_path = f"ensemble_{model_name}_best.pth"

            try:
                shutil.copy(best_path, final_path)
                selected_models.append(final_path)

                print(f"📊 {model_name}:")
                print(
                    f"   Mean: {results['mean_score']:.4f} ± {results['std_score']:.4f}"
                )
                print(
                    f"   Best: {min(results['fold_results']):.4f} (Fold {best_fold + 1})"
                )
                print(f"   Model: {final_path}")
            except Exception as e:
                print(f"⚠️ Failed to copy best model for {model_name}: {e}")

        # 앙상블 정보 저장
        if selected_models:
            ensemble_info = {
                "models": [
                    config
                    for config in config.ensemble_models
                    if any(config["name"] in path for path in selected_models)
                ],
                "model_paths": selected_models,
                "results": all_results,
                "num_classes": num_classes,
                "class_names": temp_dataset.classes,
            }

            try:
                torch.save(ensemble_info, "ensemble_info.pth")
                print("\n💾 Ensemble info saved: ensemble_info.pth")
                print("🎯 Ready for ensemble inference!")
            except Exception as e:
                print(f"❌ Failed to save ensemble info: {e}")
        else:
            print("❌ No models were successfully trained!")
    else:
        print("❌ No models completed training successfully!")


if __name__ == "__main__":
    main()
