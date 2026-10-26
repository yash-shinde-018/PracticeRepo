
# PCA Visualization using Wine Dataset

import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

print("Original dataset shape:", X.shape)

# 1. Data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Reduced dataset shape:", X_pca.shape)

# 3. Display explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("Total explained variance:",
      pca.explained_variance_ratio_.sum())

# 4. Visualize the dataset
plt.figure(figsize=(8, 6))

for i, name in enumerate(wine.target_names):
    plt.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        label=name
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of Wine Dataset")
plt.legend()
plt.grid(True)
plt.show()
