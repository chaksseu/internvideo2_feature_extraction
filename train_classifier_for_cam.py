import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random

# 시드 설정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Step 1: Dataset 정의
class VideoFeatureDataset(Dataset):
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

            return features, label
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            return None

# Custom collate_fn 정의
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    features, labels = zip(*batch)
    features = torch.stack(features, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return features, labels

# Step 2: 모델 정의 (용량 증가)
class VideoClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(VideoClassifier, self).__init__()
        #self.fc1 = nn.Linear(input_dim, 128)
        #self.relu = nn.ReLU()
        #self.fc2 = nn.Linear(128, num_classes)
        self.linear_probing = nn.Linear(input_dim, num_classes)

    def forward(self, x):  # x.shape = (batch_size, T, D)
        x = x.mean(dim=1)  # 시간 축 평균 (batch_size, D)
        x = self.linear_probing(x)
        #x = self.fc1(x)
        #x = self.relu(x)
        #x = self.fc2(x)
        return x

# Step 3: 학습 함수 정의
def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs, val_loader=None, device='cpu'):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            if batch is None:  # 건너뛰기
                continue
            features, labels = batch
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        epoch_loss = running_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # 검증 데이터로 평가
        if val_loader is not None:
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    # 학습 로그 반환
    return model, train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_model(model, dataloader, criterion, device='cpu'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:  # 건너뛰기
                continue
            features, labels = batch
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return loss, accuracy

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

if __name__ == "__main__":
    # 설정
    csv_file = "dataset_annotations.csv"
    feature_dir = "./feature_ucf_subset/processed_128_intern2_s2"
    splits = ['train', 'val', 'test']
    num_epochs = 10
    batch_size = 64  # 배치 크기 조정
    learning_rate = 0.001  # 학습률 조정
    fixed_length = 128
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 클래스 이름 추출
    df = pd.read_csv(csv_file)
    class_names = df['class'].unique()
    class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}
    num_classes = len(class_to_idx)

    # 데이터셋 및 데이터로더 생성
    datasets = {}
    for split in splits:
        datasets[split] = VideoFeatureDataset(csv_file, feature_dir, class_to_idx, split, fixed_length=fixed_length)

    
    # 데이터셋 평균과 표준편차 계산
    mean, std = compute_dataset_mean_std(datasets)

    # 데이터셋에 평균과 표준편차 설정
    for split in splits:
        datasets[split].mean = mean
        datasets[split].std = std

    dataloaders = {}
    for split in splits:
        shuffle = True if split == 'train' else False
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )

    # 입력 차원 결정
    input_dim = datasets['train'][0][0].shape[1]

    # 모델, 손실 함수, 옵티마이저, 스케줄러 정의
    model = VideoClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 모델 학습
    trained_model, train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, dataloaders['train'], criterion, optimizer, scheduler, num_epochs,
        val_loader= dataloaders['val'], device=device
    )

    # 테스트 데이터로 평가
    test_loss, test_acc = evaluate_model(trained_model, dataloaders['test'], criterion, device=device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # 모델 저장
    save_dir = "./saved_models"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(trained_model.state_dict(), os.path.join(save_dir, "video_classifier.pth"))
    print("Model training complete and saved!")