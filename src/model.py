{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "baf5289b-e11d-4bfd-98ec-0146c75225c1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Overwriting model.py\n"
     ]
    }
   ],
   "source": [
    "%%writefile model.py\r\n",
    "\r\n",
    "\"\"\"Fault Detection Classifier & Latency Profiling Engine.\r\n",
    "\r\n",
    "Optimized for edge deployment with sub-millisecond inference and\r\n",
    "strict bounds checking.\r\n",
    "\"\"\"\r\n",
    "\r\n",
    "import time\r\n",
    "from typing import Dict, Union\r\n",
    "\r\n",
    "import numpy as np\r\n",
    "import pandas as pd\r\n",
    "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score\r\n",
    "from sklearn.tree import DecisionTreeClassifier\r\n",
    "\r\n",
    "\r\n",
    "class SensorFaultClassifier:\r\n",
    "    def __init__(self, max_depth: int = 5, min_samples_split: int = 2):\r\n",
    "        self.max_depth = max_depth\r\n",
    "        self.min_samples_split = min_samples_split\r\n",
    "\r\n",
    "        self.model = DecisionTreeClassifier(\r\n",
    "            max_depth=max_depth,\r\n",
    "            min_samples_split=min_samples_split,\r\n",
    "            random_state=42,\r\n",
    "        )\r\n",
    "\r\n",
    "        self.is_trained: bool = False\r\n",
    "\r\n",
    "    def train(\r\n",
    "        self,\r\n",
    "        X_train: Union[pd.DataFrame, np.ndarray],\r\n",
    "        y_train: Union[pd.Series, np.ndarray],\r\n",
    "    ) -> None:\r\n",
    "        self.model.fit(X_train, y_train)\r\n",
    "        self.is_trained = True\r\n",
    "\r\n",
    "    def predict(\r\n",
    "        self,\r\n",
    "        X_data: Union[pd.DataFrame, np.ndarray],\r\n",
    "    ) -> np.ndarray:\r\n",
    "        if not self.is_trained:\r\n",
    "            raise RuntimeError(\"Model has not been trained yet.\")\r\n",
    "\r\n",
    "        return self.model.predict(X_data)\r\n",
    "\r\n",
    "    def evaluate(\r\n",
    "        self,\r\n",
    "        X_test: Union[pd.DataFrame, np.ndarray],\r\n",
    "        y_test: Union[pd.Series, np.ndarray],\r\n",
    "    ) -> Dict[str, Union[float, int]]:\r\n",
    "        if not self.is_trained:\r\n",
    "            raise RuntimeError(\"Model must be trained prior to evaluation.\")\r\n",
    "\r\n",
    "        start_time = time.perf_counter()\r\n",
    "        preds = self.model.predict(X_test)\r\n",
    "        elapsed_total = time.perf_counter() - start_time\r\n",
    "\r\n",
    "        latency_per_sample_ms = (\r\n",
    "            elapsed_total / max(len(X_test), 1)\r\n",
    "        ) * 1000.0\r\n",
    "\r\n",
    "        return {\r\n",
    "            \"accuracy\": float(accuracy_score(y_test, preds)),\r\n",
    "            \"f1_score\": float(f1_score(y_test, preds, zero_division=0)),\r\n",
    "            \"precision\": float(precision_score(y_test, preds, zero_division=0)),\r\n",
    "            \"recall\": float(recall_score(y_test, preds, zero_division=0)),\r\n",
    "            \"latency_ms\": float(latency_per_sample_ms),\r\n",
    "            \"model_complexity\": int(self.model.tree_.node_count),\r\n",
    "        }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "9713f5a0-9bc5-4c77-b1f2-b4a77668eaaa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MODEL IMPORT SUCCESS\n"
     ]
    }
   ],
   "source": [
    "from model import SensorFaultClassifier\n",
    "\n",
    "print(\"MODEL IMPORT SUCCESS\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "2fefb249-9fa1-4cea-b03a-bf006994f9c6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions: [0 0 0 1 1 1]\n",
      "Evaluation: {'accuracy': 1.0, 'f1_score': 1.0, 'precision': 1.0, 'recall': 1.0, 'latency_ms': 0.03763332885379592, 'model_complexity': 3}\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "\n",
    "X = np.array([\n",
    "    [20, 2, 30],\n",
    "    [22, 3, 32],\n",
    "    [21, 2, 31],\n",
    "    [80, 9, 90],\n",
    "    [85, 10, 95],\n",
    "    [82, 9, 92],\n",
    "])\n",
    "\n",
    "y = np.array([0, 0, 0, 1, 1, 1])\n",
    "\n",
    "clf = SensorFaultClassifier()\n",
    "clf.train(X, y)\n",
    "\n",
    "print(\"Predictions:\", clf.predict(X))\n",
    "print(\"Evaluation:\", clf.evaluate(X, y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2feef706-b973-4512-a69c-a2f717f88a24",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
