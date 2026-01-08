import numpy as np
from src.utils.evaluation import calculate_metrics

def test_calculate_metrics_perfect_prediction():
    """Test metrics when predictions match true labels perfectly."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    # Confusion matrix should be diagonal
    expected_cm = np.array([[2, 0], [0, 2]])
    np.testing.assert_array_equal(metrics["confusion_matrix"], expected_cm)

def test_calculate_metrics_all_wrong():
    """Test metrics when all predictions are wrong."""
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert metrics["accuracy"] == 0.0
    assert metrics["f1"] == 0.0
    
def test_calculate_metrics_custom_labels():
    """Test if the function handles custom label ordering correctly."""
    y_true = [0, 1]
    y_pred = [0, 1]
    
    # Normal call
    metrics = calculate_metrics(y_true, y_pred, labels=[0, 1])
    assert metrics["accuracy"] == 1.0

def test_calculate_metrics_zero_division():
    """Test stability when one class is never predicted (zero division)."""
    y_true = [0, 0, 0]
    y_pred = [0, 0, 0]
    
    # This typically causes ZeroDivisionError in raw calculation if not handled
    metrics = calculate_metrics(y_true, y_pred)
    
    assert metrics["accuracy"] == 1.0
    # Precision/Recall for class 1 will be 0, macro average will reflect that
    # We just ensure it doesn't crash
    assert isinstance(metrics["f1"], float)
