import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Step 1: Dataset 정의
class VideoFeatureDataset(Dataset):
    def __init__(self, csv_file, feature_dir, class_to_idx, split):
        """
        Args:
            csv_file (str): CSV 파일 경로
            feature_dir (str): .npy 파일이 저장된 폴더 경로
            class_to_idx (dict): 클래스 이름을 정수 레이블로 매핑한 딕셔너리
            split (str): 'train', 'val', 'test' 중 하나
        """
        self.feature_dir = feature_dir
        self.class_to_idx = class_to_idx
        self.data = []

        # CSV 파일 로드
        df = pd.read_csv(csv_file)

        # 원하는 split만 선택
        df = df[df['split'] == split]

        for idx, row in df.iterrows():
            class_name = row['class']
            filename = row['filename']
            npy_path = os.path.join(feature_dir, filename.replace('.avi', '.npy'))
            label = class_to_idx[class_name]
            if os.path.exists(npy_path):
                self.data.append((npy_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_path, label = self.data[idx]
        features = np.load(npy_path)  # Shape: (T_i, D)
        # 시간 축에 대해 평균 풀링 적용
        features = features.mean(axis=0)  # Shape: (D,)
        return torch.tensor(features, dtype=torch.float32), label

# Step 2: 모델 정의
class VideoClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(VideoClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)

# Step 3: 학습 함수 정의
def train_model(model, dataloader, criterion, optimizer, num_epochs, val_loader=None):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in dataloader:
            features, labels = features.cuda(), labels.cuda()

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

        epoch_loss = running_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # 검증 데이터로 평가
        if val_loader is not None:
            val_loss, val_acc = evaluate_model(model, val_loader, criterion)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
    return model

def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.cuda(), labels.cuda()

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    model.train()
    return loss, accuracy

# Step 4: 데이터 준비 및 학습 실행
if __name__ == "__main__":
    # 설정
    csv_file = "dataset_annotations.csv"  # CSV 파일 경로
    feature_dir = "./feature_ucf_subset_2s/th14_vit_g_16_4"  # npy 파일들이 저장된 경로
    splits = ['train', 'val', 'test']
    num_epochs = 1000
    batch_size = 64
    learning_rate = 0.005

    # 클래스 이름 추출
    df = pd.read_csv(csv_file)
    class_names = df['class'].unique()
    class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}
    num_classes = len(class_to_idx)

    # 데이터셋 및 데이터로더 생성
    datasets = {}
    dataloaders = {}
    for split in splits:
        datasets[split] = VideoFeatureDataset(csv_file, feature_dir, class_to_idx, split)
        shuffle = True if split == 'train' else False
        dataloaders[split] = DataLoader(datasets[split], batch_size=batch_size, shuffle=shuffle)

    # 입력 차원 결정 (첫 번째 샘플 사용)
    sample_features, _ = datasets['train'][0]
    input_dim = sample_features.shape[0]

    # 모델, 손실 함수, 옵티마이저 정의
    model = VideoClassifier(input_dim, num_classes).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 모델 학습
    trained_model = train_model(model, dataloaders['train'], criterion, optimizer, num_epochs, val_loader=dataloaders['val'])

    # 테스트 데이터로 평가
    test_loss, test_acc = evaluate_model(trained_model, dataloaders['test'], criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # 모델 저장
    torch.save(trained_model.state_dict(), "video_classifier.pth")
    print("Model training complete and saved!")
