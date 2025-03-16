import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from preprocessing.feature_extraction import BagOfWords
from classifiers.minimum_distance_classifier import MinimumDistanceClassifier
from sklearn.metrics import accuracy_score


DATASET_FILENAME = 'PhiUSIIL_Phishing_URL_Dataset.csv'
DATASET_FILE = os.path.join('..', 'dataset', DATASET_FILENAME)


def main():
    dataset = pd.read_csv(DATASET_FILE)
    # TODO performance metrics (ROC, accuracy, precision, F1, etc)
    # TODO implement some code to not use only libs

    # 80 training, 20 test # TODO implement K Cross Validation
    test_data, train_data = train_test_split(dataset, test_size=0.8, random_state=42)
    train_data_labels = train_data['label']
    train_data.drop(columns=["FILENAME", "URL", "Domain", "TLD", "Title", "label"], inplace=True)
    test_data_labels = test_data['label']
    test_data.drop(columns=["FILENAME", "URL", "Domain", "TLD", "Title", "label"], inplace=True)

    '''
    # remove non numeric features TODO represent these features as numeric (BoW maybe)
    dataset.drop(columns=["FILENAME", "URL", "Domain", "TLD", "Title", "label"], inplace=True)
    # calculate features correlation matrix
    correlation_matrix = dataset.corr()
    plt.figure(figsize=(40, 32))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.savefig('correlation_matrix.png')
    plt.show()
    '''

    lda = LinearDiscriminantAnalysis(n_components=1)
    X_train_lda = lda.fit_transform(train_data, train_data_labels)
    X_test_lda = lda.transform(test_data)

    plt.figure(figsize=(8, 6))
    sns.histplot(x=X_train_lda.flatten(), hue=train_data_labels, palette="viridis", kde=True, element="step")
    plt.title("LDA Transformation (1 Component)")
    plt.xlabel("LDA Component")
    plt.legend(title="Class")
    plt.show()

    mdc_raw = MinimumDistanceClassifier()
    mdc_raw.fit(train_data, train_data_labels)
    predictions = mdc_raw.predict(test_data)
    accuracy = accuracy_score(predictions, test_data_labels)
    print("Accuracy without data preprocessing:", accuracy)

    mdc_lda = MinimumDistanceClassifier()
    mdc_lda.fit(X_train_lda, train_data_labels)
    predictions = mdc_lda.predict(X_test_lda)
    accuracy = accuracy_score(predictions, test_data_labels)
    print("Accuracy using LDA:", accuracy)

    #bow = BagOfWords(list(dataset['Title']))

if __name__ == '__main__':
    main()