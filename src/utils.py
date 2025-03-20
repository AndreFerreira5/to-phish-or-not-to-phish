import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    RocCurveDisplay,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def display_predictions_performance(
        predictions,
        test_data_labels,
        title="Model Performance"
):
    TN, FP, FN, TP = confusion_matrix(test_data_labels, predictions).ravel()
    print(f"----------{title}----------")
    print("Precision:", precision_score(test_data_labels, predictions))
    print("Recall:", recall_score(test_data_labels, predictions))
    print("F1:", f1_score(test_data_labels, predictions))
    print("Accuracy:", accuracy_score(test_data_labels, predictions))
    print("ROC AUC:", roc_auc_score(test_data_labels, predictions))
    print("TP:", TP)
    print("FP:", FP)
    print("TN:", TN)
    print("FN:", FN)
    print("-"*(len(title)+20))

    RocCurveDisplay.from_predictions(test_data_labels, predictions)
    plt.show()


def display_folding_predictions_performance(
        predictions: list,
        test_data_labels: list,
        title=f"Unknown Model Performance"
):
    fold_num = len(predictions)
    print(f"----------{title}----------")
    print("Fold\tPrecision\tRecall\tF1\tAccuracy\tROC AUC\tTP\tFP\tTN\tFN")

    precision_sum = 0
    recall_sum = 0
    f1_sum = 0
    accuracy_sum = 0
    roc_auc_sum = 0
    TP_sum = 0
    FN_sum = 0
    TN_sum = 0
    FP_sum = 0
    #folding_metrics = {"precision": [], "recall": [], "f1": [], "accuracy": [], "roc_auc": [], "tp": [], "fp": [], "tn": [], "fn": []}
    for i, (pred, test_labels) in enumerate(zip(predictions, test_data_labels)):
        precision = precision_score(test_labels, pred)
        recall = recall_score(test_labels, pred)
        f1 = f1_score(test_labels, pred)
        accuracy = accuracy_score(test_labels, pred)
        ROC_AUC = roc_auc_score(test_labels, pred)
        TN, FP, FN, TP = confusion_matrix(test_labels, pred).ravel()

        print(f"{i}\t{precision}\t{recall}\t{f1}\t{accuracy}\t{ROC_AUC}\t{TP}\t{FP}\t{TN}\t{FN}")
        #RocCurveDisplay.from_predictions(test_labels, pred)

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        accuracy_sum += accuracy
        roc_auc_sum += ROC_AUC
        TP_sum += TP
        FN_sum += FN
        TN_sum += TN
        FP_sum += FP

    print(f"Average\t{precision_sum/fold_num}\t{recall_sum/fold_num}\t{f1_sum/fold_num}\t{accuracy_sum/fold_num}\t{roc_auc_sum/fold_num}\t{TP_sum/fold_num}\t{FP_sum/fold_num}\t{TN_sum/fold_num}\t{FN_sum/fold_num}")
    print("-"*(len(title)+20))


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