import numpy as np

# 簡單的非隨機 image 和 mask
image = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
mask = np.array([
    [1, 0, 2],
    [2, 3, 3],
    [1, 2, 0]
])

# 計算每個 mask 值的加總和計數
sums = np.bincount(mask.flatten(), weights=image.flatten(), minlength=256)
counts = np.bincount(mask.flatten(), minlength=256)

print(sums, counts)

# 避免除以零，計算平均值
averages = np.zeros_like(sums)
nonzero = counts > 0
averages[nonzero] = sums[nonzero] / counts[nonzero]

# 將平均值映射回原圖
result = averages[mask]
result[mask == 0] = image[mask == 0]

print("Image:")
print(image)
print("\nMask:")
print(mask)
print("\nResult:")
print(result)

pad_reflect = np.pad(image, pad_width=1, mode='reflect')
print("\nPad Reflect:")
print(pad_reflect)

pad_edge = np.pad(image, pad_width=1, mode='edge')
print("\nPad Edge:")
print(pad_edge)



