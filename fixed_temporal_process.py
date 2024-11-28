import os
import numpy as np

def preprocess_and_save_features(input_dir, output_dir, fixed_length=128):
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.npy'):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)
                
                try:
                    # 데이터 로드
                    features = np.load(input_path)  # Shape: (T, D)
                    
                    # 길이 고정
                    if features.shape[0] < fixed_length:
                        # 패딩 추가
                        pad_length = fixed_length - features.shape[0]
                        padding = np.zeros((pad_length, features.shape[1]))
                        features = np.vstack((features, padding))
                    elif features.shape[0] > fixed_length:
                        # 잘라내기
                        features = features[:fixed_length, :]
                    
                    # 변환된 데이터 저장
                    np.save(output_path, features)
                    print(f"Processed and saved: {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")

# 입력 및 출력 디렉토리 설정
input_dir = "./feature_ucf_subset/intern2_s2"
output_dir = "./feature_ucf_subset/processed_128_intern2_s2"

# 길이 고정 및 저장 실행
preprocess_and_save_features(input_dir, output_dir, fixed_length=128)
