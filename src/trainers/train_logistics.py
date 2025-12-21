import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data.data_loader import CEDDataset
from data.feature_engineer import FeatureEngineer

def main():
    rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datapath = os.path.join(rootpath, r'data//CED_Dataset')

    loader = CEDDataset(datapath)
    extractor = FeatureEngineer()

    # load data
    dataset = loader.load_all()
    # split dataset
    train_set, test_set, y_train, y_test = loader.split_dataset(dataset)

    # extract features
    X_train = np.array([extractor.extract_features(event) for event in train_set])
    X_test = np.array([extractor.extract_features(event) for event in test_set])

    # standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train model
    model = LogisticRegression(random_state=42)
    print("Training...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print("Accuracy:", acc)
    print("F1 score:", f1)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()