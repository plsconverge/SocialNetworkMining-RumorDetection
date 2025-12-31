import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
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
    print("Extracting features...")
    X_train = np.array([extractor.extract_features(event) for event in train_set])
    X_test = np.array([extractor.extract_features(event) for event in test_set])
    
  
    selected_features_idx = [8, 16] # likes, followers
    
    
    X_train_selected = X_train[:, selected_features_idx]
    X_test_selected = X_test[:, selected_features_idx]
    
    # standardize all features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # standardize selected features
    scaler_selected = StandardScaler()
    X_train_selected_scaled = scaler_selected.fit_transform(X_train_selected)
    X_test_selected_scaled = scaler_selected.transform(X_test_selected)
    
    print(f"\n{'='*80}")
    print("=== Model Comparison ===")
    print(f"{'='*80}\n")
    
    # 1. all features
    print("1. Original Model (no class_weight) + All Features")
    model1 = LogisticRegression(random_state=42, max_iter=1000)
    model1.fit(X_train_scaled, y_train)
    y_pred1 = model1.predict(X_test_scaled)
    acc1 = accuracy_score(y_test, y_pred1)
    f1_1 = f1_score(y_test, y_pred1, average='macro')
    print(f"   Accuracy: {acc1:.4f}, F1: {f1_1:.4f}")
    print("   Classification report:")
    print(classification_report(y_test, y_pred1, target_names=['Non-Rumor', 'Rumor']))
    
    # 2. selected features
    print("\n2. Original Model (no class_weight) + Selected Features")
    model2 = LogisticRegression(random_state=42, max_iter=1000)
    model2.fit(X_train_selected_scaled, y_train)
    y_pred2 = model2.predict(X_test_selected_scaled)
    acc2 = accuracy_score(y_test, y_pred2)
    f1_2 = f1_score(y_test, y_pred2, average='macro')
    print(f"   Accuracy: {acc2:.4f}, F1: {f1_2:.4f}")
    print("   Classification report:")
    print(classification_report(y_test, y_pred2, target_names=['Non-Rumor', 'Rumor']))
     
    # summary
    print(f"\n{'='*80}")
    print("=== Summary ===")
    print(f"{'='*80}")
    print(f"{'Model':<50} {'Accuracy':<12} {'F1 Score':<12}")
    print("-" * 80)
    print(f"{'1. All Features':<50} {acc1:<12.4f} {f1_1:<12.4f}")
    print(f"{'2. Selected Features':<50} {acc2:<12.4f} {f1_2:<12.4f}")


if __name__ == '__main__':
    main()