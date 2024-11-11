import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# 生成控制點
def generate_control_points(num_curves=3, points_per_curve=50, noise_level=0.02):
    control_points = []
    for i in range(num_curves):
        u = np.linspace(0, 1, points_per_curve)
        v = 0.3 * np.sin(2 * np.pi * u) + i * 0.3  # 用正弦波生成曲線
        u += np.random.normal(0, noise_level, points_per_curve)  # 添加少量 u 軸的隨機偏移
        v += np.random.normal(0, noise_level, points_per_curve)  # 添加少量 v 軸的隨機偏移
        curve_points = np.stack([u, v], axis=1)
        control_points.append(curve_points)
    return np.concatenate(control_points)

# 生成並可視化控制點
points = generate_control_points()

# 使用 DBSCAN 聚類點，根據 y 值分層
clustering = DBSCAN(eps=0.01, min_samples=5)
labels = clustering.fit_predict(points[:, 1].reshape(-1, 1))

# 計算每層的目標高度，讓層次分佈均勻
unique_labels = np.unique(labels[labels >= 0])
target_y_positions = np.linspace(0, 1, len(unique_labels))

# 將每層的點垂直平移到目標高度
transformed_points = points.copy()
for i, label in enumerate(unique_labels):
    layer_points = points[labels == label]
    current_mean_y = np.mean(layer_points[:, 1])
    target_y = target_y_positions[i]
    shift_amount = target_y - current_mean_y
    transformed_points[labels == label, 1] += shift_amount

# 視覺化原始點和轉換後的點
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')
axes[0].set_title("Original Control Points")

axes[1].scatter(transformed_points[:, 0], transformed_points[:, 1], c=labels, cmap='viridis')
axes[1].set_title("Transformed Control Points")

plt.show()

