import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms

from config import Config
from model import ImprovedModel
from car_dataset import CarDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()

    # 수정: transform 추가 (필수!)
    val_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = CarDataset(config.train_root, transform=None)
    class_names = full_dataset.classes

    # 수정: transform 적용
    test_dataset = CarDataset(config.test_root, transform=val_transform, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # 수정: 일관된 모델 사용
    model = ImprovedModel(num_classes=len(class_names), model_name="efficientnet_b3")

    # 수정: 모델 로딩 방식 개선
    checkpoint = torch.load("best_model_fold_0.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    model.eval()
    results = []
    filenames = []  # 수정: 파일명 저장

    with torch.no_grad():
        for images, file_names in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)

            for i, prob in enumerate(probabilities.cpu()):
                result = {
                    class_names[j]: prob[j].item() for j in range(len(class_names))
                }
                results.append(result)
                filenames.append(file_names[i])  # 파일명 저장

    predictions = pd.DataFrame(results)
    predictions.insert(0, "filename", filenames)  # 파일명 컬럼 추가

    # 수정: submission 파일 처리 개선
    try:
        submission = pd.read_csv("sample_submission.csv", encoding="utf-8-sig")
        class_columns = submission.columns[1:]

        # 파일명 기준으로 정렬/매칭
        predictions = predictions.set_index("filename")
        submission = submission.set_index(submission.columns[0])

        submission[class_columns] = predictions[class_columns]
        submission.reset_index().to_csv(
            "submission.csv", index=False, encoding="utf-8-sig"
        )

    except FileNotFoundError:
        # sample_submission.csv가 없는 경우
        predictions.to_csv("submission.csv", index=False, encoding="utf-8-sig")

    print("Submission file created: submission.csv")
