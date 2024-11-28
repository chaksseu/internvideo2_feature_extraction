import torch

# [1, 3, 8] 모양의 데이터 생성 (1부터 24까지 채움)
input_data2 = torch.arange(1, 1 + 1 * 3 * 8).view(1, 3, 8)

# Reshape: [2, 3, 4]
input_data3 = input_data2.reshape(2, 3, 4)

# 결과 확인
print("Original shape:", input_data2.shape)
print("Reshaped shape:", input_data3.shape)

# 데이터 확인
print("\nOriginal data:\n", input_data2)
print("\nFirst batch (batch 0):\n", input_data3[0])  # 첫 번째 배치
print("\nSecond batch (batch 1):\n", input_data3[1])  # 두 번째 배치
