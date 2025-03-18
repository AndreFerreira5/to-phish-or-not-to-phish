import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    RocCurveDisplay
)


def display_predictions_performance(predictions, test_data_labels, title=""):
    print(f"----------{title}----------")
    print("Precision:", precision_score(test_data_labels, predictions))
    print("Recall:", recall_score(test_data_labels, predictions))
    print("F1:", f1_score(test_data_labels, predictions))
    print("Accuracy:", accuracy_score(test_data_labels, predictions))
    print("ROC AUC:", roc_auc_score(test_data_labels, predictions))
    print("-"*(len(title)+20))

    RocCurveDisplay.from_predictions(test_data_labels, predictions)
    plt.show()
