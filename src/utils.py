import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    RocCurveDisplay
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def display_predictions_performance(
        predictions,
        test_data_labels,
        title="Model Performance"
):
    print(f"----------{title}----------")
    print("Precision:", precision_score(test_data_labels, predictions))
    print("Recall:", recall_score(test_data_labels, predictions))
    print("F1:", f1_score(test_data_labels, predictions))
    print("Accuracy:", accuracy_score(test_data_labels, predictions))
    print("ROC AUC:", roc_auc_score(test_data_labels, predictions))
    print("-"*(len(title)+20))

    RocCurveDisplay.from_predictions(test_data_labels, predictions)
    plt.show()


def plot_feature_correlation_matrix(
        dataset,
        save_plot=True,
        filename="correlation_matrix.png",
        x=80,
        y=64
):
    correlation_matrix = dataset.corr()
    plt.figure(figsize=(x,y))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.savefig(filename)
    plt.show()


def process_non_numeric_data(dataset):
    dataset = dataset.copy()

    label_encoders = {}
    for col in ["TLD", "Domain"]:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])
        label_encoders[col] = le

    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=100)
    url_tfidf = tfidf.fit_transform(dataset["URL"])
    title_tfidf = tfidf.fit_transform(dataset["Title"])

    url_tfidf_df = pd.DataFrame(url_tfidf.toarray(), columns=[f"URL_{i}" for i in range(url_tfidf.shape[1])])
    title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=[f"Title_{i}" for i in range(url_tfidf.shape[1])])

    dataset.drop(columns=["URL", "Title"], inplace=True)
    dataset = pd.concat([dataset, url_tfidf_df, title_tfidf_df], axis=1)

    dataset["File_Extension"] = dataset["FILENAME"].apply(lambda x: x.split('.')[-1])
    le = LabelEncoder()
    dataset["File_Extension"] = le.fit_transform(dataset["File_Extension"])
    dataset.drop(columns=["FILENAME"], inplace=True)

    return dataset