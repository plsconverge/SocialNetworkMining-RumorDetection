import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


from data.data_loader import CEDDataset
from data.feature_engineer import FeatureEngineer


def main():
    rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datapath = os.path.join(rootpath, r'data//CED_Dataset')

    loader = CEDDataset(datapath)
    extractor = FeatureEngineer()

    # load data
    dataset = loader.load_all()
    # split
    train_set, test_set, y_train, y_test = loader.split_dataset(dataset)

    # extract features
    print("Extracting features...")
    features_train = [extractor.extract_features_advanced(event) for event in train_set]
    features_test = [extractor.extract_features_advanced(event) for event in test_set]

    X_train = extractor.convert_to_dataframe(features_train)
    X_test = extractor.convert_to_dataframe(features_test)

    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    print("Training...")
    rf = RandomForestClassifier(max_depth=15, random_state=42)
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    y_test_pred = 1 - y_test_pred

    print(f"Train ratio: {sum(y_train[0]) / len(y_train)}")
    print(f"Test ratio: {sum(y_test[0]) / len(y_test)}")
    print(f"Train prediction ratio: {sum(y_train_pred) / len(y_train_pred)}")
    print(f"Test prediction ratio: {sum(y_test_pred) / len(y_test_pred)}")

    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1': f1_score(y_train, y_train_pred)
    }
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred)
    }

    print("Train set metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("Test set metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()