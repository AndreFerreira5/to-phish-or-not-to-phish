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
    process_non_numeric_data
)


DATASET_FILENAME = 'PhiUSIIL_Phishing_URL_Dataset.csv'
DATASET_FILE = os.path.join('..', 'dataset', DATASET_FILENAME)
USE_NON_NUMERIC_FEATURES = False
SCALE_DATA = True
PLOT_CORRELATION_MATRIX = False
PLOT_KRUSKALWALLIS_TEST_FEATURES = False
USE_KRUSKALWALLIS = True
NUMBER_OF_FEATURES = 13

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

    # Apply Kruskal-Wallis Test for feature selection
    kruskal_test = KruskalWallisTest(clean_dataset)
    results = kruskal_test.perform_test(SKIP_FEATURES=USE_NON_NUMERIC_FEATURES)
    kruskal_test.print_results(results)

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Plot the Kruskal-Wallis test results for all features
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[feature for feature, _ in sorted_results], y=[h for _, h in sorted_results])
    plt.title("Kruskal-Wallis Test Results - Feature Discrimination (H Statistic)")
    plt.xlabel("Feature")
    plt.ylabel("H Statistic")
    plt.xticks(rotation=90)
    plt.show()

    # Assuming 'results' is a list of tuples (feature_name, H_statistic)
    h_statistics = [h for _, h in results]

    # Calculate the Standard Deviation (Desvio Padrão)
    std_dev = np.std(h_statistics)

    # Calculate the Quartiles
    Q1 = np.percentile(h_statistics, 25)
    Q2 = np.percentile(h_statistics, 50)  # This is the median
    Q3 = np.percentile(h_statistics, 75)

    # Print the results
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"First Quartile (Q1): {Q1:.2f}")
    print(f"Median (Q2): {Q2:.2f}")
    print(f"Third Quartile (Q3): {Q3:.2f}")

    # Create a boxplot to visualize the quartiles, median, and outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=h_statistics)
    plt.title("Distribution of Kruskal-Wallis Test H Statistics")
    plt.xlabel("H Statistic")
    plt.show()

    # Select top 13 features based on Kruskal-Wallis significance
    top_features = [feature for feature, _ in results[:NUMBER_OF_FEATURES]]  # Select top  features



    # Plot the top 13 features based on Kruskal-Wallis significance
    if PLOT_KRUSKALWALLIS_TEST_FEATURES:
        feature_plotter = FeaturePlotter(clean_dataset, results)
        feature_plotter.plot_features(top_n=NUMBER_OF_FEATURES)

    # Plot correlation matrix of Kruskal-Wallis results
    selected_features_data = clean_dataset[top_features]  # Dataset with the selected top features
    plot_feature_correlation_matrix(selected_features_data,x=10,y=8)


    if USE_NON_NUMERIC_FEATURES:
        clean_dataset = processed_dataset
        clean_dataset.to_csv("processed_dataset.csv", index=False)
    else:
        clean_dataset = dataset
        clean_dataset.drop(columns=["label"], inplace=True)
        

    X = clean_dataset.to_numpy()

    if PLOT_CORRELATION_MATRIX:
        plot_feature_correlation_matrix(clean_dataset)

    # - Data Normalization -

    if SCALE_DATA:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        selected_features_data = scaler.fit_transform(selected_features_data)


    # - Kfold -
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # RAW Data + Euclidean MDC
    predictions = []
    test_indexes = []
    for i, (train_index, test_index) in enumerate(kfold.split(X)):
        mdc_raw = EuclideanMinimumDistanceClassifier()
        mdc_raw.fit(X[train_index], y[train_index])
        predictions.append(mdc_raw.predict(X[test_index]))
        test_indexes.append(y[test_index])
    display_folding_predictions_performance(
        predictions,
        test_indexes,
        title=f"RAW Data + Euclidean Minimum Distance Classifier"
    )

    # RAW Data + Mahalanobis MDC
    predictions = []
    test_indexes = []
    for i, (train_index, test_index) in enumerate(kfold.split(X)):
        mdc_raw = MahalanobisMinimumDistanceClassifier()
        mdc_raw.fit(X[train_index], y[train_index])
        predictions.append(mdc_raw.predict(X[test_index]))
        test_indexes.append(y[test_index])
    display_folding_predictions_performance(
        predictions,
        test_indexes,
        title=f"RAW Data + Mahalanobis Minimum Distance Classifier"
    )

    # LDA
    lda = LinearDiscriminantAnalysis(n_components=1)

    if USE_KRUSKALWALLIS:
        X_LDA = lda.fit_transform(selected_features_data, y)  # This uses only Kruskal Features
    else:
        X_LDA = lda.fit_transform(X, y)  # This uses all features


    plt.figure(figsize=(8, 6))
    sns.histplot(x=X_LDA.flatten(), hue=y, palette="viridis", kde=True, element="step")
    plt.title("LDA Transformation (1 Component)")
    plt.xlabel("LDA Component")
    plt.legend(title="Class")
    plt.show()

    # LDA + Euclidean MDC
    predictions = []
    test_indexes = []
    for i, (train_index, test_index) in enumerate(kfold.split(X_LDA)):
        mdc = EuclideanMinimumDistanceClassifier()
        mdc.fit(X_LDA[train_index], y[train_index])
        predictions.append(mdc.predict(X_LDA[test_index]))
        test_indexes.append(y[test_index])
    display_folding_predictions_performance(
        predictions,
        test_indexes,
        title=f"LDA + Euclidean Minimum Distance Classifier"
    )

    # - PCA -
    pca = PCA(n_components= 0.95)

    if USE_KRUSKALWALLIS:
        X_pca = pca.fit_transform(selected_features_data)  # this uses kruskal top feautres
    else:
        X_pca = pca.fit_transform(X)  # this uses all feautres

    # Eigenvalues (explained variance ratios)
    eigenvalues = pca.explained_variance_

    # Scree Plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--')
    plt.title("Scree Plot")
    plt.xlabel("Principal Components")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.show()

    # Apply the Kaiser Criterion: Select components with eigenvalue > 1
    kaiser_components = [i for i, ev in enumerate(eigenvalues) if ev > 1]
    print(f"Number of components based on Kaiser Criterion (eigenvalue > 1): {len(kaiser_components)}")


    # Plot the explained variance ratio for PCA
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.title("Explained Variance Ratio for Each PCA Component")
    plt.xlabel("PCA Component")
    plt.ylabel("Explained Variance Ratio")
    plt.show()

    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs Number of Components')
    plt.show()

    # PCA + Euclidean MDC
    predictions = []
    test_indexes = []

    # Print the number of components selected and explained variance ratio
    print(f"Number of components selected: {pca.n_components_}")
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

    # Test Euclidean Minimum Distance Classifier
    for i, (train_index, test_index) in enumerate(kfold.split(X_pca)):
        mdc_euclidean = EuclideanMinimumDistanceClassifier()
        mdc_euclidean.fit(X_pca[train_index], y[train_index])
        predictions.append(mdc_euclidean.predict(X_pca[test_index]))
        test_indexes.append(y[test_index])
    display_folding_predictions_performance(
        predictions,
        test_indexes,
        title=f"PCA + Euclidean Minimum Distance Classifier"
    )


    # PCA + Mahalanobis MDC
    predictions = []
    test_indexes = []
    for i, (train_index, test_index) in enumerate(kfold.split(X_pca)):
        print(X_pca[train_index].shape)
        mdc_mahalanobis = MahalanobisMinimumDistanceClassifier()
        mdc_mahalanobis.fit(X_pca[train_index], y[train_index])
        predictions.append(mdc_mahalanobis.predict(X_pca[test_index]))
        test_indexes.append(y[test_index])
    display_folding_predictions_performance(
        predictions,
        test_indexes,
        title=f"PCA + Mahalanobis Minimum Distance Classifier"
    )





if __name__ == '__main__':
    main()