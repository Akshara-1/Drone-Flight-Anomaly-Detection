import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# 1. Enhanced Data Generation
def generate_telemetry_data(n_points=1000, anomaly_rate=0.01):
    """
    Generates a time-series dataset with realistic drone telemetry.
    Includes normal and anomalous data points.
    """
    time = np.arange(n_points)
    
    # Normal flight path data
    altitude = 100 + 5 * np.sin(time / 20) + np.random.normal(0, 0.5, n_points)
    velocity = 10 + np.random.normal(0, 0.2, n_points)
    yaw = 0.2 * time + np.random.normal(0, 0.5, n_points)
    pitch = np.random.normal(2, 0.1, n_points) # Assuming slight variations
    roll = np.random.normal(0, 0.2, n_points)  # Assuming roll is near zero for stable flight
    battery = 100 - (time * 0.05) + np.random.normal(0, 0.5, n_points) # Linear battery drain
    gps_drift = np.random.normal(0, 0.1, n_points)
    
    # Create the DataFrame
    data = pd.DataFrame({
        'time': time,
        'altitude': altitude,
        'velocity': velocity,
        'yaw': yaw,
        'pitch': pitch,
        'roll': roll,
        'battery': battery,
        'gps_drift': gps_drift
    })
    
    # Add anomalies based on the specified rate
    n_anomalies = int(n_points * anomaly_rate)
    anomaly_indices = np.random.choice(time, n_anomalies, replace=False)
    
    for i in anomaly_indices:
        anomaly_type = np.random.choice(['altitude_drop', 'gps_jamming', 'yaw_spike', 'battery_drop'])
        if anomaly_type == 'altitude_drop':
            data.loc[i, 'altitude'] -= np.random.uniform(20, 50)
        elif anomaly_type == 'gps_jamming':
            data.loc[i, 'gps_drift'] += np.random.uniform(5, 15)
        elif anomaly_type == 'yaw_spike':
            data.loc[i, 'yaw'] += np.random.uniform(30, 80)
        elif anomaly_type == 'battery_drop':
            data.loc[i, 'battery'] -= np.random.uniform(10, 20)
            
    return data

# Generate a single dataset for this demo
full_data = generate_telemetry_data(n_points=1500, anomaly_rate=0.02)
features = ['altitude', 'velocity', 'yaw', 'pitch', 'roll', 'battery', 'gps_drift']

print("Sample of generated data with anomalies:")
print(full_data.head())
print("\n")

# 2. Build and Train the Model using a Pipeline
# A pipeline combines scaling and the model into a single object for cleaner code
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LocalOutlierFactor(n_neighbors=20, contamination='auto'))
])

# The LOF model is trained on the entire dataset. It identifies points that
# are "local" outliers based on their neighborhood density.
pipeline.fit(full_data[features])

print("LOF Model trained successfully.")

# 3. Predict Anomalies and Visualize
# The `fit_predict` method of LOF returns 1 for inliers and -1 for outliers
full_data['is_anomaly'] = pipeline.named_steps['model'].fit_predict(pipeline.named_steps['scaler'].transform(full_data[features]))
full_data['is_anomaly'] = full_data['is_anomaly'] == -1

print("\nAnomalies detected:")
anomalies = full_data[full_data['is_anomaly']]
print(anomalies)

# Plotting the results
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 12))
fig.suptitle('Drone Flight Anomaly Detection (Local Outlier Factor)', fontsize=16)

# Subplot 1: Altitude
axes[0, 0].set_title('Altitude')
axes[0, 0].plot(full_data['time'], full_data['altitude'], label='Altitude')
axes[0, 0].scatter(anomalies['time'], anomalies['altitude'], color='red', s=50, label='Anomaly')
axes[0, 0].legend()

# Subplot 2: Velocity
axes[0, 1].set_title('Velocity')
axes[0, 1].plot(full_data['time'], full_data['velocity'], label='Velocity')
axes[0, 1].scatter(anomalies['time'], anomalies['velocity'], color='red', s=50, label='Anomaly')
axes[0, 1].legend()

# Subplot 3: Yaw
axes[1, 0].set_title('Yaw')
axes[1, 0].plot(full_data['time'], full_data['yaw'], label='Yaw')
axes[1, 0].scatter(anomalies['time'], anomalies['yaw'], color='red', s=50, label='Anomaly')
axes[1, 0].legend()

# Subplot 4: GPS Drift
axes[1, 1].set_title('GPS Drift')
axes[1, 1].plot(full_data['time'], full_data['gps_drift'], label='GPS Drift')
axes[1, 1].scatter(anomalies['time'], anomalies['gps_drift'], color='red', s=50, label='Anomaly')
axes[1, 1].legend()

# Subplot 5: Battery
axes[2, 0].set_title('Battery')
axes[2, 0].plot(full_data['time'], full_data['battery'], label='Battery')
axes[2, 0].scatter(anomalies['time'], anomalies['battery'], color='red', s=50, label='Anomaly')
axes[2, 0].legend()

# Subplot 6: Pitch
axes[2, 1].set_title('Pitch')
axes[2, 1].plot(full_data['time'], full_data['pitch'], label='Pitch')
axes[2, 1].scatter(anomalies['time'], anomalies['pitch'], color='red', s=50, label='Anomaly')
axes[2, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("\n")
