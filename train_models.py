# ============================================================
# PREDICTING HIT SONGS USING REPEATED CHORUS
# MACHINE LEARNING MODEL TRAINING
# ============================================================

import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "chorus_features_csv.csv"

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_SPLITS = 5

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = "model_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 75)
print("LOADING DATASET")
print("=" * 75)

df = pd.read_csv(DATASET, encoding="cp1252")

print(f"Dataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 75)
print("PREPARING FEATURES")
print("=" * 75)

# track_name is metadata and must NOT be used as a feature
feature_columns = [
    column
    for column in df.columns
    if column not in ["track_name", "is_hit"]
]

X = df[feature_columns]

y = df["is_hit"]

print(f"Total audio features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")

# ============================================================
# CHECK CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

print(
    y.value_counts()
    .sort_index()
    .rename(index={
        0: "Non-Hit",
        1: "Hit"
    })
)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 75)
print("TRAIN / TEST SPLIT")
print("=" * 75)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())

# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# COMMON PREPROCESSING
# ============================================================

# IMPORTANT:
#
# Everything below is inside the Pipeline.
#
# Therefore:
#
# VarianceThreshold
#       ↓
# StandardScaler
#       ↓
# PCA
#
# are fitted ONLY on the training portion of each CV fold.
#
# This prevents data leakage.

def create_pca_pipeline(model):
    pipeline = Pipeline([

        # Remove features with zero variance
        (
            "variance_filter",
            VarianceThreshold(threshold=0.0)
        ),

        # Standardize audio features
        (
            "scaler",
            StandardScaler()
        ),

        # Retain 95% of variance
        (
            "pca",
            PCA(
                n_components=0.95,
                svd_solver="full"
            )
        ),

        # Machine learning model
        (
            "model",
            model
        )
    ])

    return pipeline


# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    # --------------------------------------------------------
    # 1. LOGISTIC REGRESSION
    # --------------------------------------------------------

    "Logistic Regression": {

        "pipeline": create_pca_pipeline(
            LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE
            )
        ),

        "parameters": {

            "model__C": [
                0.01,
                0.1,
                1,
                10,
                100
            ],

            "model__penalty": [
                "l2"
            ],

            "model__solver": [
                "lbfgs"
            ]
        }
    },

    # --------------------------------------------------------
    # 2. LINEAR SVM
    # --------------------------------------------------------

    "Linear SVM": {

        "pipeline": create_pca_pipeline(
            SVC(
                kernel="linear",
                random_state=RANDOM_STATE
            )
        ),

        "parameters": {

            "model__C": [
                0.01,
                0.1,
                1,
                10,
                100
            ]
        }
    },

    # --------------------------------------------------------
    # 3. RBF SVM
    # --------------------------------------------------------

    "RBF SVM": {

        "pipeline": create_pca_pipeline(
            SVC(
                kernel="rbf",
                random_state=RANDOM_STATE
            )
        ),

        "parameters": {

            "model__C": [
                0.1,
                1,
                10,
                100
            ],

            "model__gamma": [
                "scale",
                0.001,
                0.01,
                0.1
            ]
        }
    },

    # --------------------------------------------------------
    # 4. LDA
    # --------------------------------------------------------

    "LDA": {

        "pipeline": create_pca_pipeline(
            LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto"
            )
        ),

        "parameters": {}
    },

    # --------------------------------------------------------
    # 5. RANDOM FOREST
    # --------------------------------------------------------

    "Random Forest": {

        "pipeline": create_pca_pipeline(
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),

        "parameters": {

            "model__n_estimators": [
                100,
                200,
                300
            ],

            "model__max_depth": [
                None,
                5,
                10,
                20
            ],

            "model__min_samples_split": [
                2,
                5
            ]
        }
    },

    # --------------------------------------------------------
    # 6. GRADIENT BOOSTING
    # --------------------------------------------------------

    "Gradient Boosting": {

        "pipeline": create_pca_pipeline(
            GradientBoostingClassifier(
                random_state=RANDOM_STATE
            )
        ),

        "parameters": {

            "model__n_estimators": [
                50,
                100,
                200
            ],

            "model__learning_rate": [
                0.01,
                0.05,
                0.1
            ],

            "model__max_depth": [
                1,
                2,
                3
            ]
        }
    },

    # --------------------------------------------------------
    # 7. NEURAL NETWORK
    # --------------------------------------------------------

    "Neural Network": {

        "pipeline": create_pca_pipeline(
            MLPClassifier(
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=RANDOM_STATE
            )
        ),

        "parameters": {

            "model__hidden_layer_sizes": [
                (32,),
                (64,),
                (128,),
                (64, 32)
            ],

            "model__alpha": [
                0.0001,
                0.001,
                0.01
            ],

            "model__learning_rate_init": [
                0.001,
                0.01
            ]
        }
    }
}

# ============================================================
# TRAIN MODELS
# ============================================================

results = []

best_models = {}

for model_name, configuration in models.items():

    print("\n")
    print("=" * 75)
    print(f"TRAINING: {model_name}")
    print("=" * 75)

    pipeline = configuration["pipeline"]

    parameters = configuration["parameters"]

    # --------------------------------------------------------
    # GRID SEARCH
    # --------------------------------------------------------

    if parameters:

        grid_search = GridSearchCV(

            estimator=pipeline,

            param_grid=parameters,

            scoring="f1",

            cv=cv,

            n_jobs=-1,

            verbose=1,

            return_train_score=True
        )

    else:

        grid_search = GridSearchCV(

            estimator=pipeline,

            param_grid=[{}],

            scoring="f1",

            cv=cv,

            n_jobs=-1,

            verbose=1,

            return_train_score=True
        )

    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    grid_search.fit(X_train, y_train)

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_model = grid_search.best_estimator_

    best_models[model_name] = best_model

    print("\nBest parameters:")

    print(grid_search.best_params_)

    print(
        f"\nBest CV F1-score: "
        f"{grid_search.best_score_:.4f}"
    )

    # --------------------------------------------------------
    # TEST SET PREDICTION
    # --------------------------------------------------------

    y_pred = best_model.predict(X_test)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({

        "Model": model_name,

        "CV_F1": grid_search.best_score_,

        "Test_Accuracy": accuracy,

        "Test_Precision": precision,

        "Test_Recall": recall,

        "Test_F1": f1

    })

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nTest Set Results:")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Non-Hit",
                "Hit"
            ],
            zero_division=0
        )
    )

# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Test_F1",
    ascending=False
)

# ============================================================
# PRINT FINAL MODEL COMPARISON
# ============================================================

print("\n\n")
print("=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print(
    f"\nModel comparison saved to: "
    f"{results_file}"
)

# ============================================================
# IDENTIFY BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = best_models[
    best_model_name
]

print("\n" + "=" * 75)
print("BEST MODEL")
print("=" * 75)

print(
    f"Best model based on test F1-score: "
    f"{best_model_name}"
)

best_row = results_df.iloc[0]

print(
    f"Accuracy : {best_row['Test_Accuracy']:.4f}"
)

print(
    f"Precision: {best_row['Test_Precision']:.4f}"
)

print(
    f"Recall   : {best_row['Test_Recall']:.4f}"
)

print(
    f"F1-score : {best_row['Test_F1']:.4f}"
)

# ============================================================
# BEST MODEL PREDICTIONS
# ============================================================

best_predictions = best_model.predict(
    X_test
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)

print("\n" + "=" * 75)
print("CONFUSION MATRIX")
print("=" * 75)

print(cm)

plt.figure(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Non-Hit",
        "Hit"
    ]
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

confusion_file = os.path.join(
    OUTPUT_DIR,
    "best_model_confusion_matrix.png"
)

plt.savefig(
    confusion_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# MODEL COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(12, 7))

x = np.arange(
    len(results_df)
)

width = 0.2

plt.bar(
    x - 1.5 * width,
    results_df["Test_Accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x - 0.5 * width,
    results_df["Test_Precision"],
    width,
    label="Precision"
)

plt.bar(
    x + 0.5 * width,
    results_df["Test_Recall"],
    width,
    label="Recall"
)

plt.bar(
    x + 1.5 * width,
    results_df["Test_F1"],
    width,
    label="F1"
)

plt.xticks(
    x,
    results_df["Model"],
    rotation=30,
    ha="right"
)

plt.ylabel("Score")

plt.xlabel("Machine Learning Model")

plt.title(
    "Model Performance Comparison"
)

plt.ylim(
    0,
    1.05
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

comparison_plot = os.path.join(
    OUTPUT_DIR,
    "model_comparison.png"
)

plt.savefig(
    comparison_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# DETERMINE PCA COMPONENTS FROM BEST MODEL
# ============================================================

print("\n" + "=" * 75)
print("PCA INFORMATION")
print("=" * 75)

pca_step = best_model.named_steps["pca"]

print(
    f"Original features: {X.shape[1]}"
)

print(
    f"PCA components used: "
    f"{pca_step.n_components_}"
)

print(
    f"Variance retained: "
    f"{pca_step.explained_variance_ratio_.sum():.4f}"
)

# ============================================================
# SAVE BEST MODEL
# ============================================================

model_file = os.path.join(
    OUTPUT_DIR,
    "best_hit_song_model.pkl"
)

joblib.dump(
    best_model,
    model_file
)

print(
    f"\nBest model saved to: "
    f"{model_file}"
)

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": best_predictions

})

prediction_file = os.path.join(
    OUTPUT_DIR,
    "best_model_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print(
    f"Test predictions saved to: "
    f"{prediction_file}"
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("TRAINING COMPLETE")
print("=" * 75)

print(
    f"""
Dataset:
    200 songs
    518 original audio features
    100 Hit
    100 Non-Hit

Train/Test Split:
    Training: {len(X_train)}
    Testing : {len(X_test)}

Best Model:
    {best_model_name}

PCA:
    {pca_step.n_components_} components
    {pca_step.explained_variance_ratio_.sum() * 100:.2f}% variance retained

Performance:
    Accuracy : {best_row['Test_Accuracy']:.4f}
    Precision: {best_row['Test_Precision']:.4f}
    Recall   : {best_row['Test_Recall']:.4f}
    F1-score : {best_row['Test_F1']:.4f}

Files generated:
    model_results/model_comparison.csv
    model_results/model_comparison.png
    model_results/best_model_confusion_matrix.png
    model_results/best_hit_song_model.pkl
    model_results/best_model_predictions.csv
"""
)

print("=" * 75)