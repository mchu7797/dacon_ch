import os
import pandas as pd
import torch
from tqdm import tqdm
import pickle

from config import get_config
from datasets import CarDataset
from models import get_models, BrandModel
from transforms import get_val_transform, get_tta_transforms



def load_brand_model(config, device):
    """브랜드 분류 모델 로드"""
    brand_model_path = config.brand_model_path
    brand_info_path = config.brand_info_path

    if not os.path.exists(brand_model_path) or not os.path.exists(brand_info_path):
        print("⚠️  브랜드 모델 파일이 없습니다. 브랜드 예측 생략.")
        return None, None

    # 브랜드 정보 로드
    with open(brand_info_path, "rb") as f:
        brand_info = pickle.load(f)

    num_brands = len(brand_info["brands"])
    print(f"🏷️  브랜드 수: {num_brands}")
    print(f"🏷️  브랜드 목록: {brand_info['brands']}")

    # 브랜드 모델 생성 및 로드
    brand_model = BrandModel(num_brands).to(device)

    # 가중치 로드
    brand_model.load_state_dict(torch.load(brand_model_path, map_location="cpu"))
    brand_model.eval()

    print(f"✅ 브랜드 모델 로드 완료: {brand_model_path}")
    return brand_model, brand_info


def generate_test_brand_predictions(brand_model, brand_info, test_dataset, device):
    """테스트 데이터에 대한 브랜드 예측 생성"""
    if brand_model is None:
        return {}

    brand_model = brand_model.to(device)
    brand_model.eval()

    # 테스트 데이터 로더 생성
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,  # 브랜드 예측용으로 큰 배치 사용
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    brand_predictions = {}

    print("🔮 테스트 데이터 브랜드 예측 생성 중...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(
            tqdm(test_loader, desc="Generating brand predictions")
        ):
            # 테스트 데이터는 이미지만 있음
            if test_dataset.tta_transforms is not None:
                # TTA 사용 시: [batch_size, num_tta, channels, height, width]
                images = batch_data
                images = images.mean(dim=1)  # TTA 이미지들의 평균
            else:
                # 일반적인 경우: [batch_size, channels, height, width]
                images = batch_data

            images = images.to(device)
            outputs = brand_model(images)
            probabilities = torch.softmax(outputs, dim=1)

            # 배치의 파일명들 가져오기
            batch_start = batch_idx * test_loader.batch_size
            batch_end = min(batch_start + test_loader.batch_size, len(test_dataset))

            for i in range(batch_end - batch_start):
                # 테스트 데이터의 파일 경로에서 파일명 추출
                img_path = test_dataset.data[batch_start + i]
                filename = os.path.basename(img_path)

                # 브랜드 확률 저장
                brand_predictions[filename] = probabilities[i].cpu().numpy()

    print(f"✅ 테스트 브랜드 예측 완료: {len(brand_predictions)}개")
    return brand_predictions


def predict(models, images, brand_features=None):
    """모델 앙상블 예측"""
    predictions = []

    for model in models:
        model.eval()
        with torch.no_grad():
            if (
                brand_features is not None
                and hasattr(model, "forward")
                and "brand_logits" in model.forward.__code__.co_varnames
            ):
                # AdvancedModel - 브랜드 정보 사용
                output = model(images, brand_features)
            else:
                # 일반 모델
                output = model(images)
            predictions.append(torch.softmax(output, dim=1))

    # 앙상블 평균
    ensemble_prediction = torch.stack(predictions).mean(dim=0)
    return ensemble_prediction


def predict_with_tta(models, images, brand_features=None):
    """TTA를 사용한 예측"""
    tta_predictions = []

    # TTA 이미지들에 대해 예측
    for i in range(images.shape[1]):  # TTA 차원
        tta_image = images[:, i]  # [batch_size, channels, height, width]
        prediction = predict(models, tta_image, brand_features)
        tta_predictions.append(prediction)

    # TTA 결과 평균
    return torch.stack(tta_predictions).mean(dim=0)


def evaluate():
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 클래스 이름을 가져오기 위해 전체 데이터셋을 로드합니다.
    full_dataset = CarDataset(
        config.train_directory,
        brand_predictions_path=config.brand_predictions_path,
        brand_info_path=config.brand_info_path,
        transform=None,
    )
    class_names = full_dataset.classes
    has_brand_features = full_dataset.brand_classes is not None

    print(f"Number of classes: {len(class_names)}")
    if has_brand_features:
        print(f"Number of brand classes: {len(full_dataset.brand_classes)}")

    # 브랜드 모델 로드
    brand_model, brand_info = load_brand_model(config, device)

    # 테스트 데이터셋 로드
    if config.use_tta:
        tta_transform = get_tta_transforms(config.image_size, config.mean, config.std)
        test_dataset = CarDataset(
            config.test_directory,
            brand_predictions_path=config.brand_predictions_path,
            brand_info_path=config.brand_info_path,
            tta_transforms=tta_transform,
            is_test=True,
        )
    else:
        test_dataset = CarDataset(
            config.test_directory,
            brand_predictions_path=config.brand_predictions_path,
            brand_info_path=config.brand_info_path,
            transform=get_val_transform(config.image_size, config.mean, config.std),
            is_test=True,
        )

    print(f"Test dataset size: {len(test_dataset)}")

    # 테스트 데이터에 대한 브랜드 예측 생성
    test_brand_predictions = generate_test_brand_predictions(
        brand_model, brand_info, test_dataset, device
    )

    # 테스트 데이터셋에 브랜드 예측 추가
    if test_brand_predictions:
        test_dataset.brand_predictions.update(test_brand_predictions)
        print("✅ 테스트 데이터 브랜드 예측 업데이트 완료")

    # 메인 모델들 로드
    models = get_models(
        num_classes=len(class_names),
        brand_classes=len(full_dataset.brand_classes)
        if full_dataset.brand_classes
        else None,
    )

    # 모델 가중치 로드
    for i, model in enumerate(models):
        model_path = os.path.join(
            config.model_directory, f"best_{i}.pth"
        )
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Model {i} loaded from {model_path}")
        else:
            print(
                f"⚠️  Model {i} weights not found at {model_path}"
            )

        model.to(device)
        model.eval()

    # 테스트 데이터 로더
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size
        // (config.tta_batch_size_divisor if config.use_tta else 1),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 예측 수행
    results = []
    print("🔮 모델 예측 시작...")

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Evaluating"):
            if config.use_tta:
                # TTA 사용 시
                if isinstance(batch_data, tuple) and len(batch_data) == 2:
                    images, brand_features = batch_data
                    images = images.to(device)
                    brand_features = brand_features.to(device)
                    probabilities = predict_with_tta(models, images, brand_features)
                else:
                    # batch_data가 텐서인지 확인
                    if isinstance(batch_data, torch.Tensor):
                        images = batch_data.to(device)
                    else:
                        # 리스트나 다른 형태인 경우 처리
                        images = torch.stack(batch_data).to(device) if isinstance(batch_data, list) else batch_data
                        images = images.to(device)
                    probabilities = predict_with_tta(models, images)
            else:
                # TTA 미사용 시
                if isinstance(batch_data, tuple) and len(batch_data) == 2:
                    images, brand_features = batch_data
                    images = images.to(device)
                    brand_features = brand_features.to(device)
                    probabilities = predict(models, images, brand_features)
                else:
                    # batch_data가 텐서인지 확인
                    if isinstance(batch_data, torch.Tensor):
                        images = batch_data.to(device)
                    else:
                        # 리스트나 다른 형태인 경우 처리
                        images = torch.stack(batch_data).to(device) if isinstance(batch_data, list) else batch_data
                        images = images.to(device)
                    probabilities = predict(models, images)

            for prob in probabilities.cpu():
                result = {
                    class_names[i]: prob[i].item() for i in range(len(class_names))
                }
                results.append(result)
    
    # 결과를 DataFrame으로 변환
    predictions = pd.DataFrame(results)

    # sample_submission 파일 로드
    submission = pd.read_csv("sample_submission.csv", encoding="utf-8-sig")

    # 클래스 순서 맞추기
    class_columns = submission.columns[1:]
    predictions = predictions[class_columns]

    # submission 파일에 예측 결과 저장
    submission[class_columns] = predictions.values
    submission.to_csv("submission.csv", index=False, encoding="utf-8-sig")

    print("✅ Submission file created: submission.csv")

    # 파일 크기 확인
    file_size = os.path.getsize("submission.csv") / (1024 * 1024)
    print(f"📁 Submission file size: {file_size:.1f}MB")

    if file_size < 25:
        print("⚠️  파일 크기가 작습니다. 데이터 누락 가능성을 확인하세요.")


if __name__ == "__main__":
    evaluate()
