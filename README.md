# Metaminds: Adaptive Sensor Fault Detection Engine

### Track 04: Machine Learning & Artificial Intelligence (IEEE CIS)
An edge-optimized fault detection framework built using multi-objective evolutionary computation (Genetic Algorithms) to balance classification accuracy, inference latency, and model complexity under environmental distribution shifts.

## Architecture
- `src/data_generator.py`: Generates 3-axis synthetic telemetry with calibration drift and fault injection.
- `src/model.py`: Lightweight Decision Tree classifier with per-sample latency profiling.
- `src/optimizer.py`: Evolutionary multi-objective search finding Pareto-optimal configurations.
- `tests/test_pipeline.py`: Comprehensive test fixtures validating bounds and error states.
- `app.py`: Interactive Streamlit dashboard for real-time telemetry inspection.

## Quickstart
```bash
pip install -r requirements.txt
pytest -v
streamlit run app.py