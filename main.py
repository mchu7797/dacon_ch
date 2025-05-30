import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from tqdm import tqdm

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
):
    best_logloss = float("inf")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(
            train_loader, desc=f"[Epoch {epoch + 1}/{config.epochs}] Training"
        ):
            images, labels = images.to(device_name), labels.to(device_name)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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

                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
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

        if val_logloss < best_logloss:
            best_logloss = val_logloss
            torch.save(
                model.state_dict(),
                f"{config.model_directory}/best_{model.__class__.__name__}.pth",
            )
            print(f"Best model saved with Log Loss: {best_logloss:.4f}")


def main():
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fix_random_seed(config.seed)

    full_dataset = CarDataset(config.train_directory, transform=None)
    labels = [label for _, label in full_dataset.data]
    class_names = full_dataset.classes

    show_dataset_info(full_dataset)
    print(f"Number of classes: {len(class_names)}")

    train_idx, val_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=labels,
        random_state=config.seed,
    )

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
        indices=train_idx,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    models = get_models(model_classes=len(class_names))

    for model in models:
        model.to(device)
        print(f"Training model: {model.__class__.__name__}")
        train_model(config, model, class_names, train_loader, val_loader, device)


if __name__ == "__main__":
    main()
