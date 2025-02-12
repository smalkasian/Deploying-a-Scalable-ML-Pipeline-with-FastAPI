import pytest
import numpy as np
from ml.model import train_model, inference, compute_model_metrics
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def sample_data():
    """ Generates small sample data for testing """
    X_train = np.array([[1, 0, 0, 20], [0, 1, 0, 45], [0, 0, 1, 30]])
    y_train = np.array([0, 1, 0])
    return X_train, y_train

def test_train_model(sample_data):
    """
    Test that train_model() returns a trained model and not None.
    """
    X_train, y_train = sample_data
    model = train_model(X_train, y_train)
    
    assert model is not None, "Model training failed!"
    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier!"

def test_inference(sample_data):
    """
    Test that inference() produces predictions of correct shape.
    """
    X_train, y_train = sample_data
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)

    assert preds.shape == y_train.shape, "Inference output shape mismatch!"
    assert all(p in [0, 1] for p in preds), "Predictions should be binary (0 or 1)!"

def test_compute_model_metrics():
    """
    Test that compute_model_metrics() correctly calculates precision, recall, and F1.
    """
    y_true = np.array([0, 1, 1, 0, 1])
    y_preds = np.array([0, 1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)

    assert 0 <= precision <= 1, "Precision should be between 0 and 1!"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1!"
    assert 0 <= fbeta <= 1, "F1-score should be between 0 and 1!"
