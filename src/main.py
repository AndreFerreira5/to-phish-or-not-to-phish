import os

DATASET_FILENAME = 'PhiUSIIL_Phishing_URL_Dataset.csv'
DATASET_FILE = os.path.join('..', 'dataset', DATASET_FILENAME)

with open(DATASET_FILE) as csv_file:
    for line in csv_file:
        print(line)