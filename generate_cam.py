import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd

# 시드 설정
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Dataset 정의
class VideoFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, feature_dir, class_to_idx, split, fixed_length=48, mean=None, std=None):
        self.feature_dir = feature_dir
        self.class_to_idx = class_to_idx
        self.fixed_length = fixed_length
        self.data = []

        df = pd.read_csv(csv_file)
        df = df[df['split'] == split].reset_index(drop=True)

        for idx, row in df.iterrows():
            class_name = row['class']
            filename = row['filename']
            npy_path = os.path.join(feature_dir, filename.replace('.avi', '.npy'))
            label = class_to_idx[class_name]
            if os.path.exists(npy_path):
                self.data.append((npy_path, label))
            else:
                print(f"File not found: {npy_path}")

        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_path, label = self.data[idx]
        try:
            features = np.load(npy_path)  # Shape: (T_i, D)

            # 시퀀스 길이를 self.fixed_length로 고정
            if features.shape[0] < self.fixed_length:
                pad_length = self.fixed_length - features.shape[0]
                padding = np.zeros((pad_length, features.shape[1]))
                features = np.vstack((features, padding))
            elif features.shape[0] > self.fixed_length:
                features = features[:self.fixed_length, :]

            features = torch.tensor(features, dtype=torch.float32)  # (fixed_length, D)

            # 데이터 정규화 (데이터셋 전체 평균과 표준편차 사용)
            if self.mean is not None and self.std is not None:
                features = (features - self.mean) / (self.std + 1e-5)

            return features, label, npy_path  # npy_path를 함께 반환
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            return None

# Custom collate_fn 정의
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    features, labels, npy_paths = zip(*batch)
    features = torch.stack(features, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return features, labels, npy_paths

# 모델 정의
class VideoClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(VideoClassifier, self).__init__()
        self.linear_probing = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):  # x.shape = (batch_size, T, D)
        x = x.mean(dim=1)  # 시간 축 평균 (batch_size, D)
        x = self.linear_probing(x)
        return x

# CAM 생성 함수
def generate_cam(features, weights):
    """
    Args:
        features: (T, D) 입력 특징
        weights: (D,) 모델 가중치
    Returns:
        cam: (T,) 크기의 시간축별 CAM 값
    """
    cam = torch.matmul(features, weights)  # (T,)
    cam = cam - cam.min()  # Normalize to [0, 1]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()

# CAM 시각화 및 저장 함수
def save_cam_images(cam, feature_filename, output_dir, class_name):
    """
    Args:
        cam: (T,) 크기의 CAM 값
        feature_filename: 원본 feature 파일 이름
        output_dir: CAM 이미지를 저장할 디렉토리
        class_name: 클래스 이름
    """
    plt.figure(figsize=(10, 2))
    plt.imshow(
        cam[np.newaxis, :],
        cmap='viridis',
        aspect='auto',
        extent=[0, cam.shape[0], 0, 1],
        interpolation='bilinear'
    )
    plt.colorbar(label="Activation")
    plt.title(f"Class: {class_name}, Feature: {feature_filename}")
    plt.xlabel("Time Frames")
    plt.yticks([])

    # 파일 이름에서 확장자 제거
    base_filename = os.path.basename(feature_filename).replace('.npy', '')
    save_path = os.path.join(output_dir, f"{base_filename}_{class_name}_cam.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved CAM image to {save_path}")

# 데이터셋 전체 평균과 표준편차 계산
def compute_dataset_mean_std(datasets):
    all_features = []
    for dataset in datasets.values():
        for npy_path, _ in dataset.data:
            try:
                features = np.load(npy_path)
                all_features.append(features)
            except Exception as e:
                print(f"Error loading {npy_path}: {e}")
    all_features = np.concatenate(all_features, axis=0)
    mean = torch.tensor(all_features.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(all_features.std(axis=0), dtype=torch.float32)
    return mean, std

# 모델 로드 및 CAM 생성 실행
if __name__ == "__main__":
    # 설정
    csv_file = "dataset_annotations.csv"
    feature_dir = "./feature_ucf_subset_all/processed_128_intern2_s1"
    split = 'train'
    fixed_length = 128
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 클래스 이름 매핑 로드
    df = pd.read_csv(csv_file)
    class_names = df['class'].unique()
    class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}
    num_classes = len(class_to_idx)

    # 데이터셋 및 데이터로더 생성
    datasets = {}
    datasets[split] = VideoFeatureDataset(csv_file, feature_dir, class_to_idx, split, fixed_length=fixed_length)

    # 데이터셋 평균과 표준편차 계산
    mean, std = compute_dataset_mean_std(datasets)
    datasets[split].mean = mean
    datasets[split].std = std

    dataloader = DataLoader(datasets[split], batch_size=batch_size, collate_fn=collate_fn)

    # 학습된 모델 로드
    input_dim = datasets[split][0][0].shape[1]  # features dimension
    model = VideoClassifier(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load("./saved_models/video_classifier.pth"))
    model.eval()

    # CAM 저장 경로 설정
    output_dir = "./cam_images"
    os.makedirs(output_dir, exist_ok=True)

    # CAM 생성
    for k, batch in enumerate(dataloader):
        if batch is None:
            continue
        features, labels, npy_paths = batch
        features, labels = features.to(device), labels.to(device)

        for i in range(features.size(0)):  # 배치 내 각 샘플 처리
            feature = features[i]  # (T, D)
            label = labels[i].item()
            npy_path = npy_paths[i]  # Feature file path

            # 클래스 이름 추출
            class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(label)]

            # 모델 가중치 가져오기
            weights = model.linear_probing.weight[label].detach()  # (D,)

            # CAM 생성
            cam = generate_cam(feature, weights)  # (T,)

            # CAM 이미지 저장
            save_cam_images(cam, npy_path, output_dir, class_name)
