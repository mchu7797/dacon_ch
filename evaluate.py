import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from model import CarModel
from config import get_config
from datasets import CarDataset
from transforms import get_val_transform


def main():
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 클래스 이름을 가져오기 위해 전체 데이터셋을 로드합니다.
    full_dataset = CarDataset(config.train_directory, transform=None)
    class_names = full_dataset.classes

    test_dataset = CarDataset(
        config.test_directory,
        transform=get_val_transform(config.image_size, config.mean, config.std),
        is_test=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = CarModel(num_classes=len(class_names))
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)

    model.eval()
    results = []

    with torch.no_grad():
        for images in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

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
    main()
