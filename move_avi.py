import os
import shutil

# 원본 디렉토리 및 대상 디렉토리 설정
base_dir = "./UCF101_subset"  # train, test, val 폴더가 있는 디렉토리
output_dir = "./ucf_avi"  # 비디오 파일을 모을 디렉토리

# 출력 디렉토리가 없다면 생성
os.makedirs(output_dir, exist_ok=True)

# train, test, val 디렉토리를 반복하며 작업
for split in ['train', 'test', 'val']:
    split_dir = os.path.join(base_dir, split)
    for root, dirs, files in os.walk(split_dir):
        for file in files:
            # .avi 파일만 이동
            if file.endswith(".avi"):
                source_path = os.path.join(root, file)
                target_path = os.path.join(output_dir, file)
                
                # 파일 이름 충돌 방지
                if os.path.exists(target_path):
                    base_name, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(target_path):
                        target_path = os.path.join(output_dir, f"{base_name}_{counter}{ext}")
                        counter += 1
                
                shutil.move(source_path, target_path)

print("모든 비디오 파일이 한 폴더에 정리되었습니다.")
