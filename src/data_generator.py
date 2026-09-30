import numpy as np


def generate_sensor_data(n_samples=100, seed=42):
    rng = np.random.default_rng(seed)

    # -------------------------
    # 1. Normal sensor data
    # -------------------------
    temperature = rng.normal(30, 5, n_samples)
    vibration = rng.normal(0.5, 0.1, n_samples)
    pressure = rng.normal(100, 5, n_samples)

    # Start with everything as normal
    fault = np.zeros(n_samples, dtype=int)
    fault_type = np.array(["normal"] * n_samples, dtype=object)

    # -------------------------
    # 2. Environmental shift
    # -------------------------
    temperature += 2
    vibration *= 1.05
    pressure += 3

    # -------------------------
    # 3. Spike faults
    # -------------------------
    spike_indices = rng.choice(n_samples, size=5, replace=False)

    for i in spike_indices:
        temperature[i] += 40
        vibration[i] += 0.8
        pressure[i] += 30

        fault[i] = 1
        fault_type[i] = "spike"

    # -------------------------
    # 4. Freeze faults
    # -------------------------
    freeze_start = 20
    freeze_end = 25

    temperature[freeze_start:freeze_end] = temperature[freeze_start]

    for i in range(freeze_start, freeze_end):
        fault[i] = 1
        fault_type[i] = "freeze"

    # -------------------------
    # 5. Erratic fluctuation
    # -------------------------
    erratic_indices = rng.choice(n_samples, size=5, replace=False)

    for i in erratic_indices:
        temperature[i] += rng.normal(0, 20)
        vibration[i] += rng.normal(0, 0.5)
        pressure[i] += rng.normal(0, 20)

        fault[i] = 1
        fault_type[i] = "erratic"

    return temperature, vibration, pressure, fault, fault_type


if __name__ == "__main__":

    temperature, vibration, pressure, fault, fault_type = generate_sensor_data()

    print("FINAL SENSOR DATA\n")

    for i in range(100):
        print(
            f"Temperature: {temperature[i]:.2f}, "
            f"Vibration: {vibration[i]:.2f}, "
            f"Pressure: {pressure[i]:.2f}, "
            f"Fault: {fault_type[i]}"
        )
        np.savetxt(
    "data/sensor_data.csv",
    np.column_stack(
        (temperature, vibration, pressure, fault, fault_type)
    ),
    delimiter=",",
    fmt="%s",
    header="temperature,vibration,pressure,fault,fault_type",
    comments=""
)

print("\nDataset saved to data/sensor_data.csv")