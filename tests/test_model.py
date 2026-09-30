"""
Unit tests for SensorFaultClassifier and Evolutionary Pipeline.
Evaluates data contracts, prediction bounds, latency, and untrained exception handling.
"""
import pytest
import numpy as np
import pandas as pd

# Safe import fallback for both root and src directory structures
try:
    from src.model import SensorFaultClassifier
    from src.data_generator import generate_sensor_data
except ImportError:
    from model import SensorFaultClassifier
    from data_generator import generate_sensor_data


def test_data_generation_contract():
    """Verify synthetic sensor data generates required columns with non-null values."""
    df = generate_sensor_data(n_samples=200)
    assert len(df) == 200
    assert {"temperature", "vibration", "pressure", "is_faulty"}.issubset(df.columns)
    assert not df.isnull().values.any()


def test_untrained_model_raises_runtime_error():
    """Verify inference before training triggers a strict RuntimeError exception."""
    clf = SensorFaultClassifier(max_depth=3)
    dummy_features = np.array([[65.0, 12.0, 101.0]])
    with pytest.raises(RuntimeError):
        clf.predict(dummy_features)


def test_model_training_and_evaluation_metrics():
    """Verify training runs successfully and metrics fall within standard mathematical bounds."""
    df = generate_sensor_data(n_samples=300)
    X = df[["temperature", "vibration", "pressure"]]
    y = df["is_faulty"]

    clf = SensorFaultClassifier(max_depth=4, min_samples_split=2)
    clf.train(X, y)
    metrics = clf.evaluate(X, y)

    # Validate metric bounds
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert metrics["latency_ms"] >= 0.0
    assert metrics["model_complexity"] > 0
    assert isinstance(metrics["model_complexity"], int)


def test_model_prediction_shape_and_classes():
    """Verify batch prediction returns valid binary classes (0 or 1)."""
    df = generate_sensor_data(n_samples=50)
    X = df[["temperature", "vibration", "pressure"]]
    y = df["is_faulty"]

    clf = SensorFaultClassifier(max_depth=3)
    clf.train(X, y)
    predictions = clf.predict(X)

    assert len(predictions) == 50
    assert set(predictions).issubset({0, 1})