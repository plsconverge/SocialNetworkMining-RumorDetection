import os
import sys
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import CEDDataset
from data.feature_engineer import FeatureEngineer
from utils.evaluation import calculate_metrics


def analyze_feature_importance():
    # 设置路径
    rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datapath = os.path.join(rootpath, r'data//CED_Dataset')
    
    # 参数设置
    max_depth = 3
    random_state = 42
    
    # 加载数据
    print("Loading data...")
    loader = CEDDataset(datapath)
    dataset = loader.load_all()
    train_set, test_set, y_train, y_test = loader.split_dataset(dataset)
    
    # 转换标签格式
    y_train_list = y_train if isinstance(y_train, list) else y_train.tolist() if hasattr(y_train, 'tolist') else list(y_train)
    y_test_list = y_test if isinstance(y_test, list) else y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
    
    # 提取特征
    print("Extracting features...")
    features_train = [FeatureEngineer.extract_features_advanced(event) for event in train_set]
    features_test = [FeatureEngineer.extract_features_advanced(event) for event in test_set]
    
    # 转换为DataFrame以获取特征名称
    X_train_df = FeatureEngineer.convert_to_dataframe(features_train)
    X_test_df = FeatureEngineer.convert_to_dataframe(features_test)
    
    # 获取特征名称
    feature_names = X_train_df.columns.tolist()
    
    # 转换为numpy数组用于训练
    X_train_array = X_train_df.values
    X_test_array = X_test_df.values
    
    print(f"\nData Summary:")
    print(f"- Train samples: {len(train_set)}")
    print(f"- Test samples: {len(test_set)}")
    print(f"- Number of features: {len(feature_names)}")
    print(f"- Feature names: {feature_names}")
    
    # 训练随机森林模型
    print("\nTraining Random Forest model...")
    rf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    rf.fit(X_train_array, y_train_list)
    
    # 评估模型性能
    print("\nModel Performance on Test Set:")
    y_pred = rf.predict(X_test_array)
    metrics = calculate_metrics(y_test_list, y_pred.tolist())
    print(f"- Accuracy: {metrics['accuracy']:.4f}")
    print(f"- F1 Score: {metrics['f1']:.4f}")
    print(f"- Precision: {metrics['precision']:.4f}")
    print(f"- Recall: {metrics['recall']:.4f}")
    
    # 1. 内置特征重要性
    print("\n" + "="*60)
    print("BUILT-IN FEATURE IMPORTANCE")
    print("="*60)
    print(f"{'Feature':<20} {'Importance':<15}")
    print("-"*35)
    
    built_in_importance = rf.feature_importances_
    for name, importance in sorted(zip(feature_names, built_in_importance), key=lambda x: x[1], reverse=True):
        print(f"{name:<20} {importance:.6f}")
    
    # 2. 排列重要性
    print("\n" + "="*60)
    print("PERMUTATION IMPORTANCE")
    print("="*60)
    print(f"{'Feature':<20} {'Mean':<15} {'Std':<15}")
    print("-"*50)
    
    perm_importance = permutation_importance(
        rf, X_test_array, y_test_list, n_repeats=10, random_state=random_state
    )
    
    for name, importance_mean, importance_std in sorted(
        zip(feature_names, perm_importance.importances_mean, perm_importance.importances_std),
        key=lambda x: x[1], reverse=True
    ):
        print(f"{name:<20} {importance_mean:.6f} {importance_std:.6f}")
    
    # 3. 特征重要性对比
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE COMPARISON")
    print("="*60)
    print(f"{'Feature':<20} {'Built-in':<12} {'Permutation':<12}")
    print("-"*44)
    
    # 按内置重要性排序
    sorted_idx = built_in_importance.argsort()[::-1]
    for idx in sorted_idx:
        print(f"{feature_names[idx]:<20} {built_in_importance[idx]:<12.6f} {perm_importance.importances_mean[idx]:<12.6f}")
    
    # 4. 重要特征总结
    print("\n" + "="*60)
    print("IMPORTANT FEATURES SUMMARY")
    print("="*60)
    
    # 内置重要性前5
    top5_builtin = sorted(zip(feature_names, built_in_importance), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 Features by Built-in Importance:")
    for i, (name, importance) in enumerate(top5_builtin, 1):
        print(f"{i}. {name:<20} ({importance:.6f})")
    
    # 排列重要性前5
    top5_permutation = sorted(zip(feature_names, perm_importance.importances_mean), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 Features by Permutation Importance:")
    for i, (name, importance) in enumerate(top5_permutation, 1):
        print(f"{i}. {name:<20} ({importance:.6f})")
    
    # 共同重要特征
    builtin_names = [name for name, _ in top5_builtin]
    permutation_names = [name for name, _ in top5_permutation]
    common_features = set(builtin_names) & set(permutation_names)
    
    if common_features:
        print("\nCommon Important Features:")
        for feature in common_features:
            print(f"- {feature}")
    else:
        print("\nNo common important features in top 5.")


if __name__ == "__main__":
    analyze_feature_importance()
