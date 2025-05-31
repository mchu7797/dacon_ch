import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from config import get_config
from datasets import CarDataset
from models import get_models
from transforms import get_val_transform, get_tta_transforms


def load_trained_models(config, num_classes, device):
    base_models = get_models(num_classes=num_classes)
    loaded_models = []

    for model in base_models:
        model_name = model.__class__.__name__
        model_path = f"{config.model_directory}/best_{model_name}.pth"

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            loaded_models.append(model)
            print(f"Loaded model: {model_name} from {model_path}")
        else:
            model.to(device)
            model.eval()
            loaded_models.append(model)
            print(
                f"Model {model_name} not found at {model_path}, using untrained model."
            )

    return loaded_models


def predict(models, images):
    all_predictions = []

    with torch.no_grad():
        for model in models:
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            all_predictions.append(probabilities)

    ensemble_predictions = torch.mean(torch.stack(all_predictions), dim=0)
    return ensemble_predictions


def predict_with_tta(models, tta_images):
    batch_size, num_tta, _, _, _ = tta_images.shape
    all_predictions = []

    with torch.no_grad():
        for model in models:
            tta_predictions = []

            for tta_idx in range(num_tta):
                images = tta_images[:, tta_idx, :, :, :]
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                tta_predictions.append(probabilities)

            average_prediction = torch.mean(torch.stack(tta_predictions), dim=0)
            all_predictions.append(average_prediction)

    ensemble_predictions = torch.mean(torch.stack(all_predictions), dim=0)
    return ensemble_predictions



def evaluate():
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 클래스 이름을 가져오기 위해 전체 데이터셋을 로드합니다.
    full_dataset = CarDataset(config.train_directory, transform=None)
    class_names = full_dataset.classes

    if config.use_tta:
        tta_transform = get_tta_transforms(config.image_size, config.mean, config.std)
        test_dataset = CarDataset(
            config.test_directory,
            tta_transforms=tta_transform,
            is_test=True,
        )
        batch_size = max(1, config.batch_size // config.tta_batch_size_divisor)
    else:
        test_dataset = CarDataset(
            config.test_directory,
            transform=get_val_transform(config.image_size, config.mean, config.std),
            is_test=True,
        )
        batch_size = config.batch_size

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    models = load_trained_models(config, num_classes=len(class_names), device=device)
    print(f"Number of classes: {len(class_names)}")
    print(f"Number of models loaded: {len(models)}")

    results = []

    with torch.no_grad():
        for images in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)

            if config.use_tta:
                probabilities = predict_with_tta(models, images)
            else:
                probabilities = predict(models, images)

            for prob in probabilities.cpu():
                result = {
                    class_names[i]: prob[i].item() for i in range(len(class_names))
                }
                results.append(result)

    predictions = pd.DataFrame(results)

    submission = pd.read_csv("sample_submission.csv", encoding="utf-8-sig")

    class_columns = submission.columns[1:]
    predictions = predictions[class_columns]

    submission[class_columns] = predictions.values
    submission.to_csv("submission.csv", index=False, encoding="utf-8-sig")

    print("Submission file created: submission.csv")


if __name__ == "__main__":
    evaluate()
