"""
Pytest Test Suite for OptiForge Autonomous AI Auditor.
Validates data integrity, model bounds, error handling, parameter sweeps, and genetic optimization.
"""
import pytest
import numpy as np
import pandas as pd

# Safe import fallback
try:
    from src.data_generator import generate_sensor_data
    from src.model import SensorFaultClassifier
    from src.optimizer import EvolutionaryOptimizer
except ImportError:
    from data_generator import generate_sensor_data
    from model import SensorFaultClassifier
    from optimizer import EvolutionaryOptimizer


# --- 1. Data Schema & Integrity Tests ---
def test_data_shape_and_columns():
    """Validates generated sensor stream shape and column schema."""
    df = generate_sensor_data(n_samples=250)
    assert len(df) == 250
    expected_cols = {"temperature", "vibration", "pressure", "is_faulty"}
    assert expected_cols.issubset(df.columns)


def test_no_null_values_under_extreme_drift():
    """Ensures data simulator produces clean, non-null values even under severe drift."""
    df = generate_sensor_data(n_samples=200, drift_magnitude=25.0)
    assert not df.isnull().values.any()
    assert not np.isinf(df.to_numpy()).any()


# --- 2. Parameterized Distribution Shift Sweeps ---
@pytest.mark.parametrize("drift_mag", [0.0, 5.0, 15.0, 30.0])
def test_distribution_shift_sweeps(drift_mag):
    """Stress tests telemetry distribution across varying ambient drift levels."""
    df = generate_sensor_data(n_samples=100, drift_magnitude=drift_mag)
    assert len(df) == 100
    assert df["temperature"].mean() > 50.0  # Basic thermodynamic floor


@pytest.mark.parametrize("fault_ratio", [0.05, 0.15, 0.30])
def test_fault_ratio_calibration(fault_ratio):
    """Validates that ground-truth fault labeling respects the assigned ratio."""
    df = generate_sensor_data(n_samples=400, fault_ratio=fault_ratio)
    actual_ratio = df["is_faulty"].mean()
    assert abs(actual_ratio - fault_ratio) < 0.05


# --- 3. Model Error Handling & Boundary Condition Tests ---
def test_classifier_untrained_exception():
    """Ensures predicting with an un-fitted model strictly raises RuntimeError."""
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


def test_extreme_thermal_spike_detection():
    """Validates that massive sensor spikes (>100°C) are strictly flagged as anomalies."""
    clf = SensorFaultClassifier(max_depth=4)
    df = generate_sensor_data(n_samples=300)
    clf.train(df[["temperature", "vibration", "pressure"]], df["is_faulty"])
    
    extreme_spike = np.array([[125.0, 55.0, 101.0]])
    prediction = clf.predict(extreme_spike)
    assert prediction[0] == 1


def test_complete_sensor_dropout_detection():
    """Validates that zero-signal dropouts (0.0 reading) are classified as faults."""
    clf = SensorFaultClassifier(max_depth=4)
    df = generate_sensor_data(n_samples=300)
    clf.train(df[["temperature", "vibration", "pressure"]], df["is_faulty"])
    
    dropout_sample = np.array([[0.0, 0.0, 0.0]])
    prediction = clf.predict(dropout_sample)
    assert prediction[0] == 1


# --- 4. Evolutionary Optimizer & Pareto Convergence Tests ---
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


def test_fitness_scalarization_penalties():
    """Ensures the fitness function strictly penalizes excessive latency and complexity."""
    opt = EvolutionaryOptimizer()
    low_comp_metrics = {"f1_score": 0.95, "model_complexity": 10, "latency_ms": 0.04}
    high_comp_metrics = {"f1_score": 0.95, "model_complexity": 200, "latency_ms": 5.0}

    score_low = opt.compute_fitness(low_comp_metrics)
    score_high = opt.compute_fitness(high_comp_metrics)
    assert score_low > score_high
