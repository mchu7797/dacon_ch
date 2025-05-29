import torch
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms
import numpy as np
import gc

from config import Config
from car_dataset import CarDataset
from model import Ensemble


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()

    print(f"🖥️ Using device: {device}")
    print("🔮 Starting ensemble inference...")

    # 앙상블 정보 로드 (예외 처리 추가)
    try:
        ensemble_info = torch.load("ensemble_info.pth", map_location=device)
        model_configs = ensemble_info["models"]
        model_paths = ensemble_info["model_paths"]
        num_classes = ensemble_info["num_classes"]
        class_names = ensemble_info["class_names"]
    except FileNotFoundError:
        print("❌ ensemble_info.pth not found. Run training first!")
        return
    except Exception as e:
        print(f"❌ Failed to load ensemble info: {e}")
        return

    print(f"📊 Ensemble models: {[config['name'] for config in model_configs]}")
    print(f"📁 Classes: {num_classes}")

    # 테스트 데이터 준비
    val_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    try:
        test_dataset = CarDataset(
            config.test_root, transform=val_transform, is_test=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
    except Exception as e:
        print(f"❌ Failed to load test dataset: {e}")
        return

    print(f"🧪 Test samples: {len(test_dataset)}")

    # 앙상블 모델 초기화 및 로드
    try:
        ensemble = Ensemble(model_configs, device)
        ensemble.load_models(model_paths, num_classes)
    except Exception as e:
        print(f"❌ Failed to load ensemble models: {e}")
        return

    # 개별 모델 예측 (선택사항 - 성능 비교용)
    print("\n📈 Individual model predictions...")
    individual_predictions = []
    for i, config in enumerate(model_configs):
        try:
            print(f"  Predicting with {config['name']}...")
            pred = ensemble.predict_single_model(test_loader, i)
            individual_predictions.append(pred)

            # 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"⚠️ Failed to predict with {config['name']}: {e}")
            continue

    # 앙상블 예측
    print("\n🎯 Ensemble prediction...")
    try:
        ensemble_predictions = ensemble.predict(test_loader)
    except Exception as e:
        print(f"❌ Failed ensemble prediction: {e}")
        return

    # 파일명 수집 (메모리 효율적 방법)
    filenames = [sample[1] for sample in test_dataset.samples]

    # 결과 데이터프레임 생성
    try:
        results = []
        for i, pred in enumerate(ensemble_predictions):
            result = {class_names[j]: pred[j] for j in range(len(class_names))}
            results.append(result)

        predictions_df = pd.DataFrame(results)
        predictions_df.insert(0, "filename", filenames)

        # 개별 모델 결과도 저장 (분석용)
        for i, (pred, config) in enumerate(zip(individual_predictions, model_configs)):
            individual_results = []
            for j, p in enumerate(pred):
                result = {class_names[k]: p[k] for k in range(len(class_names))}
                individual_results.append(result)

            individual_df = pd.DataFrame(individual_results)
            individual_df.insert(0, "filename", filenames)
            individual_df.to_csv(
                f"predictions_{config['name']}.csv", index=False, encoding="utf-8-sig"
            )
            print(f"💾 Saved individual predictions: predictions_{config['name']}.csv")

        # 최종 제출 파일 생성
        try:
            submission = pd.read_csv("sample_submission.csv", encoding="utf-8-sig")
            class_columns = submission.columns[1:]

            # 파일명 기준으로 정렬/매칭
            predictions_df = predictions_df.set_index("filename")
            submission = submission.set_index(submission.columns[0])

            submission[class_columns] = predictions_df[class_columns]
            submission.reset_index().to_csv(
                "ensemble_submission.csv", index=False, encoding="utf-8-sig"
            )

        except FileNotFoundError:
            # sample_submission.csv가 없는 경우
            predictions_df.to_csv(
                "ensemble_submission.csv", index=False, encoding="utf-8-sig"
            )

        print("\n✅ Ensemble inference completed!")
        print("📊 Files created:")
        print("   - ensemble_submission.csv (최종 제출 파일)")
        for config in model_configs:
            print(f"   - predictions_{config['name']}.csv (개별 모델 예측)")

        # 예측 통계 출력
        print("\n📈 Prediction Statistics:")
        print(f"   Total predictions: {len(ensemble_predictions)}")
        print(f"   Ensemble weights: {[f'{w:.3f}' for w in ensemble.weights]}")

        # 각 클래스별 평균 확률 출력
        print("\n📊 Average probabilities by class:")
        avg_probs = np.mean(ensemble_predictions, axis=0)
        for i, class_name in enumerate(class_names):
            print(f"   {class_name}: {avg_probs[i]:.4f}")

    except Exception as e:
        print(f"❌ Failed to save results: {e}")
        return


if __name__ == "__main__":
    main()
