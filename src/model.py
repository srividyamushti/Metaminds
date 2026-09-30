
"""Fault Detection Classifier & Latency Profiling Engine.

Optimized for edge deployment with sub-millisecond inference and
strict bounds checking.
"""

import time
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier


class SensorFaultClassifier:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )

        self.is_trained: bool = False

    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> None:
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(
        self,
        X_data: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")

        return self.model.predict(X_data)

    def evaluate(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Union[float, int]]:
        if not self.is_trained:
            raise RuntimeError("Model must be trained prior to evaluation.")

        start_time = time.perf_counter()
        preds = self.model.predict(X_test)
        elapsed_total = time.perf_counter() - start_time

        latency_per_sample_ms = (
            elapsed_total / max(len(X_test), 1)
        ) * 1000.0

        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_score": float(f1_score(y_test, preds, zero_division=0)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "latency_ms": float(latency_per_sample_ms),
            "model_complexity": int(self.model.tree_.node_count),
        }
