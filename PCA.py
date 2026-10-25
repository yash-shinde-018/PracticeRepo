
# Import required libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the built-in Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# i. Data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ii. Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Display explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Explained Variance:",
      pca.explained_variance_ratio_.sum())

# iii. Visualize the dataset
plt.figure(figsize=(8, 6))

colors = ['red', 'green', 'blue']
for i, name in enumerate(wine.target_names):
    plt.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        color=colors[i],
        label=name,
        alpha=0.7
    )

plt.xlabel('First Principal Component (PC1)')
plt.ylabel('Second Principal Component (PC2)')
plt.title('PCA Visualization of Wine Dataset')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
