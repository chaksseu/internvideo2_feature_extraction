import os
import csv

# 데이터셋 루트 경로를 정확히 설정하세요
dataset_root = "./UCF101_subset"  # 데이터셋이 있는 실제 경로로 수정하세요
splits = ["train", "test", "val"]
output_csv = "dataset_annotations.csv"

# CSV 파일 작성
with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    # CSV 헤더 작성
    writer.writerow(["split", "class", "filename", "filepath"])
    
    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.exists(split_dir):
            print(f"경로가 존재하지 않습니다: {split_dir}")
            continue
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                if file_name.endswith(".avi"):
                    file_path = os.path.join(class_dir, file_name)
                    writer.writerow([split, class_name, file_name, file_path])

print(f"CSV 파일이 생성되었습니다: {output_csv}")
