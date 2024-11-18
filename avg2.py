import numpy as np
from scipy.ndimage import convolve

def update_potential(potential, mask):
    # 定義卷積核心
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])  # 包括中心點的 3x3 卷積
    
    # 計算 counts：mask > 0 的數目
    binary_mask = (mask > 0).astype(int)
    counts = convolve(binary_mask, kernel, mode='constant', cval=0)
    counts[counts == 0] = 1  # 避免除以 0，設定為 1

    # 計算 sums：周圍 mask > 0 的 potential 值總和
    sums = convolve(potential * binary_mask, kernel, mode='constant', cval=0)

    # 更新 potential
    updated_potential = sums / counts
    updated_potential[mask <= 0] = potential[mask <= 0]  # 保留 mask <= 0 的原值

    return updated_potential

# 測試
n = 5
potential = np.random.rand(n, n)
mask = np.random.randint(0, 2, size=(n, n))  # 隨機產生 0 或 1 的 mask

new_potential = update_potential(potential, mask)
print("原始 potential:")
print(potential)
print("mask:")
print(mask)
print("更新後的 potential:")
print(new_potential)
