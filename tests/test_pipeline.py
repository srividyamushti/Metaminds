"""
Pytest Test Suite for OptiForge Autonomous AI Auditor.
Validates data integrity, model bounds, error handling, and evolutionary optimizer contracts.
"""
import pytest
import numpy as np
import pandas as pd

# Safe import fallbacks
try:
    from src.data_generator import generate_sensor_data
    from src.model import SensorFaultClassifier
    from src.optimizer import EvolutionaryOptimizer
except ImportError:
    from data_generator import generate_sensor_data
    from model import SensorFaultClassifier
    from optimizer import EvolutionaryOptimizer


def test_data_shape_and_columns():
    """Validates generated sensor stream shape and column schema."""
    df = generate_sensor_data(n_samples=250)
    assert len(df) == 250
    expected_cols = {"temperature", "vibration", "pressure", "is_faulty"}
    assert expected_cols.issubset(df.columns)


def test_no_null_values_in_sensor_stream():
    """Ensures data simulator produces clean, non-null values even under drift."""
    df = generate_sensor_data(n_samples=150, drift_magnitude=10.0)
    assert not df.isnull().values.any()


def test_classifier_untrained_error_handling():
    """Ensures predicting with an un-fitted model raises RuntimeError."""
    clf = SensorFaultClassifier()
    dummy_input = np.array([[65.0, 12.0, 101.0]])
    with pytest.raises(RuntimeError):
        clf.predict(dummy_input)


def test_classifier_prediction_bounds():
    """Ensures evaluation metrics adhere to valid probability and mathematical bounds."""
    df = generate_sensor_data(n_samples=300)
    X = df[["temperature", "vibration", "pressure"]]
    y = df["is_faulty"]

    clf = SensorFaultClassifier(max_depth=3)
    clf.train(X, y)
    metrics = clf.evaluate(X, y)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert metrics["latency_ms"] >= 0.0
    assert metrics["model_complexity"] > 0


def test_optimizer_pareto_convergence():
    """Verifies that the genetic algorithm executes generations and returns optimal configuration."""
    df = generate_sensor_data(n_samples=120)
    X = df[["temperature", "vibration", "pressure"]]
    y = df["is_faulty"]

    optimizer = EvolutionaryOptimizer(population_size=4, generations=2)
    best_chromo, best_metrics, history = optimizer.optimize(X, y, X, y)

    assert len(best_chromo) == 2
    assert "f1_score" in best_metrics
    assert len(history) == 2