from sklearn.datasets import load_sample_image
import numpy as np
import matplotlib.pyplot as plt

flower = load_sample_image('flower.jpg')
gray_flo = flower[:,:,0]/255.0
# flo_v = gray_flo.flatten()
# X = np.stack([flo_v, flo_v])

X_centered = gray_flo - np.mean(gray_flo, axis=0)
cov = np.cov(X_centered, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
# X_pca = np.dot(X_centered, eigvecs)

idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]

# pick top k components
k = 30
W = eigvecs[:, :k]

# project and reconstruct
X_pca = np.dot(X_centered, W)
X_reconstructed = np.dot(X_pca, W.T) + np.mean(gray_flo, axis=0)

# plot original vs reconstruction
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray_flo, cmap="gray")
axes[0].set_title("Original")
axes[1].imshow(X_reconstructed, cmap="gray")
axes[1].set_title(f"PCA Reconstructed (k={k})")
plt.show()
