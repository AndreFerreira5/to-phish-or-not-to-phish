import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from preprocessing.kruskal_wallis import KruskalWallisTest
from preprocessing.kruskal_wallis import FeaturePlotter

from classifiers.minimum_distance_classifier import (
    EuclideanMinimumDistanceClassifier,
    MahalanobisMinimumDistanceClassifier,
)
from utils import (
    display_predictions_performance,
    display_folding_predictions_performance,
    plot_feature_correlation_matrix,
    process_non_numeric_data,
    display_kw_quartiles
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from classifiers.neural_network_classifier import NeuralNetworkClassifier

DATASET_FILENAME = 'PhiUSIIL_Phishing_URL_Dataset.csv'
DATASET_FILE = os.path.join('..', 'dataset', DATASET_FILENAME)
USE_NON_NUMERIC_FEATURES = True
SCALE_DATA = True
PLOT_CORRELATION_MATRIX = False
PLOT_KRUSKALWALLIS_TEST_FEATURES = False
USE_KRUSKALWALLIS = False
NUMBER_OF_FEATURES = 13

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def get_feature_matrix(df, sel_cols, scale=False):
    """Return X (ndarray) after optional scaling."""
    X = df[sel_cols].to_numpy()
    if scale:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
    return X


def fit_and_predict(X_train, X_test, y_train, cfg):
    if cfg["metric"] == "euclidean":
        clf = EuclideanMinimumDistanceClassifier()

    elif cfg["metric"] == "mahalanobis":
        clf = MahalanobisMinimumDistanceClassifier()

    elif cfg["metric"] == "knn":
        clf = KNeighborsClassifier(n_neighbors=cfg["k"])

    elif cfg["metric"] == "svm":
        clf = SVC(kernel="linear", C=cfg["C"])

    elif cfg["metric"] == "nn":  # ← NEW
        clf = NeuralNetworkClassifier(**cfg["nn_params"])

    elif cfg["metric"] == "gnb":
        clf = GaussianNB()

    else:
        raise ValueError("Unknown metric")

    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def select_top_features_by_kw(df_train, k, label_column="label"):
    kw = KruskalWallisTest(df_train, label_column=label_column)
    results = kw.perform_test(SKIP_FEATURES=True)
    sorted_feats = sorted(results, key=lambda x: x[1], reverse=True)[:k]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=[feature for feature, _ in sorted_feats], y=[h for _, h in sorted_feats])
    plt.title("Kruskal-Wallis Test Results - Feature Discrimination (H Statistic)")
    plt.xlabel("Feature")
    plt.ylabel("H Statistic")
    plt.xticks(rotation=90)
    plt.show()

    # Assuming 'results' is a list of tuples (feature_name, H_statistic)
    h_statistics = [h for _, h in results]

    display_kw_quartiles(h_statistics)

    # Calculate the Standard Deviation (Desvio Padrão)
    #std_dev = np.std(h_statistics)

    # Calculate the Quartiles
    #Q1 = np.percentile(h_statistics, 25)
    #Q2 = np.percentile(h_statistics, 50)  # This is the median
    #Q3 = np.percentile(h_statistics, 75)

    # Print the results
    #print(f"Standard Deviation: {std_dev:.2f}")
    #print(f"First Quartile (Q1): {Q1:.2f}")
    #print(f"Median (Q2): {Q2:.2f}")
    #print(f"Third Quartile (Q3): {Q3:.2f}")

    # Create a boxplot to visualize the quartiles, median, and outliers
    #plt.figure(figsize=(10, 6))
    #sns.boxplot(x=h_statistics)
    #plt.title("Distribution of Kruskal-Wallis Test H Statistics")
    #plt.xlabel("H Statistic")
    #plt.show()

    #idx = [kw.feature_names.index(f) for f, _ in sorted_feats]
    return [col for col, _ in sorted_feats]


def main():
    if not os.path.isfile(DATASET_FILE):
        raise FileNotFoundError(f"Dataset file with path {DATASET_FILE} not found.")

    dataset = pd.read_csv(DATASET_FILE)
    processed_dataset = process_non_numeric_data(dataset)

    y = dataset["label"]


    if USE_NON_NUMERIC_FEATURES:
        clean_dataset = processed_dataset
        clean_dataset.to_csv("processed_dataset.csv", index=False)
    else:
        clean_dataset = dataset
        clean_dataset.drop(columns=["FILENAME", "URL", "Domain", "TLD", "Title"], inplace=True)

    label_col_index = clean_dataset.columns.get_loc("label")
    X = clean_dataset.to_numpy()

    if PLOT_CORRELATION_MATRIX:
        plot_feature_correlation_matrix(clean_dataset)

    # - Data Normalization -

    sel_cols = select_top_features_by_kw(clean_dataset, NUMBER_OF_FEATURES)
    selected_features_data = clean_dataset[sel_cols].to_numpy()

    if SCALE_DATA:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        selected_features_data = scaler.fit_transform(selected_features_data)

    # ------------------------------------------------------------
    # experiment definitions
    # ------------------------------------------------------------
    EXPERIMENTS = [
        ("raw+euclid", dict(scale=False, dims="raw", metric="euclidean")),
        ("raw+maha", dict(scale=False, dims="raw", metric="mahalanobis")),
        ("scaled+euclid", dict(scale=True, dims="raw", metric="euclidean")),
        ("scaled+maha", dict(scale=True, dims="raw", metric="mahalanobis")),
        ("lda+euclid", dict(scale=True, dims="lda", metric="euclidean")),
        ("pca+euclid", dict(scale=True, dims="pca", metric="euclidean")),
        ("pca+maha", dict(scale=True, dims="pca", metric="mahalanobis")),
        ("knn(k=7)", dict(scale=True, dims="raw", metric="knn", k=7)),
        ("svm(C=0.01)", dict(scale=True, dims="raw", metric="svm", C=0.01)),
        ("nn", dict(
            scale=True,
            dims="raw",
            metric="nn",
            nn_params=dict(
                hidden_layers=(128, 64),
                epochs=80,
                batch_size=64,
                patience=8,
                verbose=True
            )
        )),
        ("gnb", dict(
            scale=True,
            dims="raw",
            metric="gnb"
        )),
    ]

    results = {name: [] for name, _ in EXPERIMENTS}
    true_y = []

    knn_errors_per_fold = []
    svm_errors_per_fold = []

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(clean_dataset)):
        # ------- split once ------------------------------------
        df_train = clean_dataset.iloc[train_idx].copy()
        df_test = clean_dataset.iloc[test_idx].copy()

        # ------- KW feature selection (once per fold) ----------
        if USE_KRUSKALWALLIS:
            sel_cols = select_top_features_by_kw(df_train, NUMBER_OF_FEATURES)
        else:
            sel_cols = [c for c in df_train.columns if c != "label"]

        # ------- target ----------------------------------------
        y_train = df_train["label"].to_numpy()
        y_test = df_test["label"].to_numpy()
        true_y.append(y_test)

        # ------- raw / scaled feature matrices -----------------
        X_train_raw = get_feature_matrix(df_train, sel_cols, scale=False)
        X_test_raw = get_feature_matrix(df_test, sel_cols, scale=False)

        X_train_scaled = get_feature_matrix(df_train, sel_cols, scale=True)
        X_test_scaled = get_feature_matrix(df_test, sel_cols, scale=True)

        # ------- LDA (fits on scaled features) -----------------
        lda = LinearDiscriminantAnalysis(n_components=1).fit(X_train_scaled, y_train)
        X_train_lda = lda.transform(X_train_scaled)
        X_test_lda = lda.transform(X_test_scaled)

        plt.figure(figsize=(8, 6))
        sns.histplot(x=X_train_lda.flatten(),
                     hue=y_train,
                     palette="viridis",
                     kde=True,
                     element="step")
        plt.title(f"Fold {fold} – LDA component 1 (training)")
        plt.xlabel("LDA component 1")
        #plt.legend(title="Class")
        plt.show()

        # ------- PCA (fits on scaled features) -----------------
        pca_full = PCA().fit(X_train_scaled)  # full fit to get eigenvalues
        eigenvalues = pca_full.explained_variance_

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(eigenvalues) + 1),
                 eigenvalues,
                 marker="o",
                 ls="--")
        plt.axhline(1.0, color="red", ls="--", label="λ = 1")
        plt.title(f"Fold {fold} – Scree plot");
        plt.xlabel("PC");
        plt.ylabel("Eigenvalue")
        #plt.legend();
        plt.grid(True);
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
                pca_full.explained_variance_ratio_)
        plt.title(f"Fold {fold} – Variance ratio per PC")
        plt.xlabel("PC");
        plt.ylabel("Expl. variance ratio");
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
                 np.cumsum(pca_full.explained_variance_ratio_))
        plt.title(f"Fold {fold} – Cumulative explained variance")
        plt.xlabel("Number of PCs");
        plt.ylabel("Cumulative variance");
        plt.show()

        # Kaiser rule
        n_kaiser = max(1, (eigenvalues > 1).sum())
        pca = PCA(n_components=n_kaiser).fit(X_train_scaled)
        X_train_pca = pca.transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)


        # ------- run every experiment --------------------------
        for name, cfg in EXPERIMENTS:
            if cfg["dims"] == "raw":
                Xtr, Xte = (X_train_scaled if cfg["scale"] else X_train_raw,
                            X_test_scaled if cfg["scale"] else X_test_raw)
            elif cfg["dims"] == "lda":
                Xtr, Xte = X_train_lda, X_test_lda
            elif cfg["dims"] == "pca":
                Xtr, Xte = X_train_pca, X_test_pca
            else:
                raise ValueError

            preds = fit_and_predict(Xtr, Xte, y_train, cfg)
            results[name].append(preds)

    # ------------------------------------------------------------
    # aggregate and display
    # ------------------------------------------------------------
    for name in results:
        display_folding_predictions_performance(
            results[name], true_y, title=name
        )


if __name__ == '__main__':
    for SCALE_DATA in [True]:
        for USE_NON_NUMERIC_FEATURES in [True, False]:
            for USE_KRUSKALWALLIS in [True, False]:
                print(f"SCALE_DATA = {SCALE_DATA} / USE_NON_NUMERIC_FEATURES = {USE_NON_NUMERIC_FEATURES} / USE_KRUSKALWALLIS = {USE_KRUSKALWALLIS}")
                main()
