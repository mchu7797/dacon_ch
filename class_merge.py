# preprocess_data.py
import os
import shutil

def merge_class_folders(train_dir):
    """
    주어진 규칙에 따라 train 폴더 내의 클래스 폴더들을 병합합니다.
    'source' 폴더의 모든 이미지를 'destination' 폴더로 이동시키고,
    빈 'source' 폴더는 삭제합니다.
    """
    # 병합 규칙: {'source_폴더': 'destination_폴더'}
    merge_map = {
        'K5_3세대_하이브리드_2020_2022': 'K5_하이브리드_3세대_2020_2023',
        '디_올_뉴_니로_2022_2025': '디_올뉴니로_2022_2025',
        '박스터_718_2017_2024': '718_박스터_2017_2024',
        '라브4_4세대_2013_2018': 'RAV4_2016_2018',
        '라브4_5세대_2019_2024': 'RAV4_5세대_2019_2024'
    }

    print("--- 클래스 폴더 병합 시작 ---")
    for source_class, dest_class in merge_map.items():
        source_path = os.path.join(train_dir, source_class)
        dest_path = os.path.join(train_dir, dest_class)

        if not os.path.exists(source_path):
            print(f"[경고] 소스 폴더를 찾을 수 없음: '{source_path}', 건너뜁니다.")
            continue
        if not os.path.exists(dest_path):
            print(f"[경고] 대상 폴더를 찾을 수 없음: '{dest_path}', 건너뜁니다.")
            continue

        print(f"'{source_class}' -> '{dest_class}' 병합 중...")
        image_files = os.listdir(source_path)
        for fname in image_files:
            shutil.move(os.path.join(source_path, fname), os.path.join(dest_path, fname))

        # 이미지를 모두 옮긴 후 빈 소스 폴더 삭제
        os.rmdir(source_path)
        print(f"'{source_class}' 폴더 삭제 완료. 총 {len(image_files)}개 이미지 이동.")

    print("--- 모든 클래스 폴더 병합 완료 ---")

if __name__ == '__main__':
    # autorun.sh에서 train 폴더가 있는 현재 디렉토리에서 실행될 것을 가정
    TRAIN_DIRECTORY = "train"
    merge_class_folders(TRAIN_DIRECTORY)