import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# CONFIGURATION
# ============================================================

FILE = "chorus_features_csv.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FILE, encoding="cp1252")

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"Audio features: {df.shape[1] - 2}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(df["is_hit"].value_counts())

print("\nPercentages:")
print(df["is_hit"].value_counts(normalize=True) * 100)


# ============================================================
# CHECK DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes.value_counts())


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isna().sum()

if missing.sum() == 0:
    print("No missing values found.")
else:
    print(missing[missing > 0])


# ============================================================
# CHECK INFINITE VALUES
# ============================================================

print("\n" + "=" * 70)
print("INFINITE VALUES")
print("=" * 70)

feature_columns = [
    col for col in df.columns
    if col not in ["track_name", "is_hit"]
]

X = df[feature_columns]

infinite_count = np.isinf(X.values).sum()

print(f"Infinite values: {infinite_count}")


# ============================================================
# FEATURE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)

print(X.describe().T.head(20))


# ============================================================
# FEATURE VARIANCE
# ============================================================

print("\n" + "=" * 70)
print("ZERO-VARIANCE FEATURES")
print("=" * 70)

variances = X.var()

zero_variance = variances[variances == 0]

print(f"Zero-variance features: {len(zero_variance)}")

if len(zero_variance) > 0:
    print(zero_variance)


# ============================================================
# CLASS DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(7, 5))

df["is_hit"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Hit vs Non-Hit Songs")
plt.xlabel("Class (0 = Non-Hit, 1 = Hit)")
plt.ylabel("Number of Songs")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
plt.show()


# ============================================================
# STANDARDIZE FEATURES FOR PCA
# ============================================================

print("\n" + "=" * 70)
print("PCA ANALYSIS")
print("=" * 70)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ============================================================
# PCA - ALL COMPONENTS
# ============================================================

pca = PCA()

X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)


# ============================================================
# NUMBER OF COMPONENTS FOR DIFFERENT VARIANCE LEVELS
# ============================================================

for threshold in [0.90, 0.95, 0.99]:

    n_components = np.argmax(
        cumulative_variance >= threshold
    ) + 1

    print(
        f"Components for {threshold * 100:.0f}% variance: "
        f"{n_components}"
    )


# ============================================================
# PCA EXPLAINED VARIANCE PLOT
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance
)

plt.axhline(
    y=0.90,
    linestyle="--",
    label="90% variance"
)

plt.axhline(
    y=0.95,
    linestyle="--",
    label="95% variance"
)

plt.axhline(
    y=0.99,
    linestyle="--",
    label="99% variance"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("pca_explained_variance.png", dpi=300)
plt.show()


# ============================================================
# PCA 2D VISUALIZATION
# ============================================================

pca_2 = PCA(n_components=2)

X_pca_2 = pca_2.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

hit = df["is_hit"] == 1
non_hit = df["is_hit"] == 0

plt.scatter(
    X_pca_2[non_hit, 0],
    X_pca_2[non_hit, 1],
    label="Non-Hit",
    alpha=0.7
)

plt.scatter(
    X_pca_2[hit, 0],
    X_pca_2[hit, 1],
    label="Hit",
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("PCA Projection of Songs")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("pca_2d.png", dpi=300)
plt.show()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print("Generated:")
print("1. class_distribution.png")
print("2. pca_explained_variance.png")
print("3. pca_2d.png")