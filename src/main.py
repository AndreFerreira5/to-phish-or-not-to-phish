import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold

from preprocessing.feature_extraction import BagOfWords
from classifiers.minimum_distance_classifier import (
    EuclideanMinimumDistanceClassifier,
    MahalanobisMinimumDistanceClassifier,
)
from utils import (
    display_predictions_performance,
    plot_feature_correlation_matrix
)


DATASET_FILENAME = 'PhiUSIIL_Phishing_URL_Dataset.csv'
DATASET_FILE = os.path.join('..', 'dataset', DATASET_FILENAME)


def main():
    if not os.path.isfile(DATASET_FILE):
        raise FileNotFoundError(f"Dataset file with path {DATASET_FILE} not found.")

    dataset = pd.read_csv(DATASET_FILE)
    y = dataset["label"]

    clean_dataset = dataset
    clean_dataset.drop(columns=["FILENAME", "URL", "Domain", "TLD", "Title", "label"], inplace=True)
    X = clean_dataset.to_numpy()
    plot_feature_correlation_matrix(clean_dataset)

    # TODO Kruskal Wallis, Data Standardization/Normalization, PCA (and then test with euclidean MDC and mahalanobis MDC)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(kfold.split(X)):
        mdc_raw = MahalanobisMinimumDistanceClassifier()
        mdc_raw.fit(X[train_index], y[train_index])
        predictions = mdc_raw.predict(X[test_index])
        display_predictions_performance(
            predictions,
            y[test_index],
            title=f"RAW Data + Mahalanobis Minimum Distance Classifier FOLD {i}"
        )

    lda = LinearDiscriminantAnalysis(n_components=1)
    X_LDA = lda.fit_transform(X, y)
    plt.figure(figsize=(8, 6))
    sns.histplot(x=X_LDA.flatten(), hue=y, palette="viridis", kde=True, element="step")
    plt.title("LDA Transformation (1 Component)")
    plt.xlabel("LDA Component")
    plt.legend(title="Class")
    plt.show()

    for i, (train_index, test_index) in enumerate(kfold.split(X_LDA)):
        mdc = EuclideanMinimumDistanceClassifier()
        mdc.fit(X_LDA[train_index], y[train_index])
        predictions = mdc.predict(X_LDA[test_index])
        display_predictions_performance(
            predictions,
            y[test_index],
            title=f"LDA + Euclidean Minimum Distance Classifier FOLD {i}"
        )



if __name__ == '__main__':
    main()