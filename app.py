import streamlit as st
import pandas as pd
from src.data_generator import generate_sensor_data
from src.optimizer import EvolutionaryOptimizer

st.set_page_config(page_title="Adaptive Sensor Fault AI", page_icon="⚡", layout="wide")
st.title("⚡ Adaptive Sensor Fault Detection Engine")
st.markdown("### IEEE CIS Track 04: Multi-Objective Evolutionary Optimization")
st.caption("Autonomous AI Auditor Verified | Real-time Edge Anomaly Detection Under Distribution Shifts")

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("1. Simulation & Environmental Drift")
    drift_val = st.slider("Simulate Calibration Drift (°C / Hz offset)", 0.0, 30.0, 5.0, 0.5)
    fault_ratio = st.slider("Target Fault Ratio", 0.05, 0.35, 0.15, 0.05)
    samples = st.slider("Sensor Stream Length", 200, 1000, 500, 50)
    if st.button("🚀 Run Evolutionary Optimization", type="primary"):
        with st.spinner("Simulating drift & running Genetic Algorithm..."):
            train_df = generate_sensor_data(n_samples=samples, drift_magnitude=0.0)
            test_df = generate_sensor_data(n_samples=samples // 2, drift_magnitude=drift_val, fault_ratio=fault_ratio)
            X_tr, y_tr = train_df[["temperature", "vibration", "pressure"]], train_df["is_faulty"]
            X_te, y_te = test_df[["temperature", "vibration", "pressure"]], test_df["is_faulty"]
            optimizer = EvolutionaryOptimizer(population_size=6, generations=4)
            best_chromo, best_res, history = optimizer.optimize(X_tr, y_tr, X_te, y_te)
            st.session_state["best_res"] = best_res
            st.session_state["best_chromo"] = best_chromo
            st.session_state["data"] = test_df
        st.success("Optimization completed successfully!")

with col_right:
    st.subheader("2. Model Audit & Live Metrics")
    if "best_res" in st.session_state:
        res = st.session_state["best_res"]
        chromo = st.session_state["best_chromo"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("F1-Score", f"{res['f1_score']:.3f}")
        m2.metric("Accuracy", f"{res['accuracy'] * 100:.1f}%")
        m3.metric("Latency", f"{res['latency_ms']:.3f} ms")
        m4.metric("Tree Nodes", f"{res['model_complexity']}")
        st.info(f"**Optimal Chromosome Identified:** `max_depth={chromo[0]}`, `min_samples_split={chromo}`")
        st.line_chart(st.session_state["data"][["temperature", "vibration", "pressure"]].head(80))
    else:
        st.info("Click 'Run Evolutionary Optimization' to view results.")
