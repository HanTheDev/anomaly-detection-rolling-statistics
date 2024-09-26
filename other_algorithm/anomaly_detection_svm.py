import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import random
from collections import deque
from matplotlib.widgets import Slider, Button
# from playsound import playsound
import threading
import datetime
import csv
import os
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# ----------------------------- Parameters -----------------------------
WINDOW_SIZE = 200               # Number of data points in the sliding window
ANOMALY_PROBABILITY = 0.02      # Probability of injecting an anomaly
ANOMALY_MULTIPLIER = 5          # Magnitude of injected anomaly
UPDATE_INTERVAL = 100           # Milliseconds between plot updates
TRAINING_INTERVAL = 50          # Data points between retraining

# One-Class SVM Parameters
SVM_KERNEL = 'rbf'               # Kernel type
SVM_NU = 0.02                    # An upper bound on the fraction of anomalies
SVM_GAMMA = 'scale'              # Kernel coefficient

# Rolling Statistics Parameters
ROLLING_WINDOW = 50             # Window size for rolling statistics
STD_THRESHOLD = 3               # Number of standard deviations for bounds

# Initialize data storage
all_x = []                      # All indices
all_y = []                      # All values
data_window = deque(maxlen=WINDOW_SIZE)      # Data for training
x_window = deque(maxlen=WINDOW_SIZE)        # Indices for plotting
anomalies_window = deque(maxlen=WINDOW_SIZE) # Anomaly flags for plotting
index = 0
total_anomalies = 0            # Total anomalies detected
new_points_since_train = 0     # Count since last training

# Initialize the scaler
scaler = StandardScaler()

# Initialize the One-Class SVM model
model = OneClassSVM(kernel=SVM_KERNEL, nu=SVM_NU, gamma=SVM_GAMMA)
model_initialized = False

# Multivariate feature engineering (optional)
def extract_features(data_window):
    """Extracts additional features from the data window."""
    features = []
    data_list = list(data_window)
    for i in range(len(data_list)):
        if i < 4:
            # Not enough data points for moving average and std
            moving_avg = np.mean(data_list[:i+1])
            moving_std = np.std(data_list[:i+1]) if len(data_list[:i+1]) > 1 else 1.0
        else:
            moving_avg = np.mean(data_list[i-4:i+1])  # Moving average over last 5 points
            moving_std = np.std(data_list[i-4:i+1]) if len(data_list[i-4:i+1]) > 1 else 1.0
        features.append([data_list[i], moving_avg, moving_std])
    return features

# Ensure anomalies_log_svm.csv exists
if not os.path.isfile('anomalies_log_svm.csv'):
    with open('anomalies_log_svm.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Value', 'Timestamp', 'Anomaly_Type'])

# Function to generate the next data point
def generate_data_point():
    global index, total_anomalies, new_points_since_train
    index += 1
    # Generate normal data
    value = np.random.normal(loc=0.0, scale=1.0)

    # Inject anomaly based on probability
    if random.random() < ANOMALY_PROBABILITY:
        std_dev = np.std(data_window) if len(data_window) > 1 else 1.0
        value += random.choice([-1, 1]) * ANOMALY_MULTIPLIER * std_dev
        is_anomaly = True
        total_anomalies += 1

        # Log anomaly
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('anomalies_log_svm.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([index, value, timestamp, 'Injected'])

        # Play alert sound in a separate thread
        # play_alert_sound()
    else:
        is_anomaly = False

    return index, value, is_anomaly

# # Function to play alert sound
# def play_alert_sound():
#     alert_sound_path = 'alert.mp3'  # Ensure you have an alert.mp3 file in your working directory
#     if os.path.isfile(alert_sound_path):
#         threading.Thread(target=playsound, args=(alert_sound_path,), daemon=True).start()
#     else:
#         print("Alert sound file not found. Please ensure 'alert.mp3' exists in the working directory.")

# Function to update the One-Class SVM model
def update_model():
    global model, model_initialized, scaler
    if len(data_window) >= 50:  # Minimum samples to train
        # Feature extraction (uncomment if using multivariate features)
        # X = extract_features(data_window)
        # For univariate data, reshape to 2D
        X = np.array(data_window).reshape(-1, 1)
        
        # Scale the data
        X_scaled = scaler.fit_transform(X)
        
        # Retrain the model
        model = OneClassSVM(kernel=SVM_KERNEL, nu=SVM_NU, gamma=SVM_GAMMA)
        model.fit(X_scaled)
        model_initialized = True
        print(f"Model retrained at index {all_x[-1]}")


# Function to update the plot
def update_plot(frame, line, anomaly_scatter, rolling_mean_line, upper_bound_line, lower_bound_line, stats_text):
    global total_anomalies, new_points_since_train, model_initialized
    # Generate new data point
    idx, val, injected_anomaly = generate_data_point()

    # Append to all data lists
    all_x.append(idx)
    all_y.append(val)

    # Append to windowed deques
    x_window.append(idx)
    data_window.append(val)
    new_points_since_train += 1

    # Feature extraction (uncomment if using multivariate features)
    # features = extract_features(data_window)

    # Check if it's time to retrain the model
    if new_points_since_train >= TRAINING_INTERVAL:
        update_model()
        new_points_since_train = 0

    # Anomaly detection using One-Class SVM
    anomaly = False
    if model_initialized and len(data_window) >= 50:
        # Feature extraction (uncomment if using multivariate features)
        # X_new = extract_features([val])
        # X_new_scaled = scaler.transform(X_new)
        X_new = np.array([[val]])
        X_new_scaled = scaler.transform(X_new)
        prediction = model.predict(X_new_scaled)
        if prediction[0] == -1:
            anomaly = True
            total_anomalies += 1

            # Log anomaly
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open('anomalies_log_svm.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([idx, val, timestamp, 'Detected'])

            # Play alert sound
            # play_alert_sound()

    # Append anomaly flag
    anomalies_window.append(anomaly)

    # Prepare data for plotting
    y_data = list(data_window)
    x_data = list(x_window)
    anomaly_flags = list(anomalies_window)

    # Update the main line
    line.set_data(x_data, y_data)

    # Extract anomalies for marking
    anomaly_x = [x_data[i] for i in range(len(y_data)) if anomaly_flags[i]]
    anomaly_y = [y_data[i] for i in range(len(y_data)) if anomaly_flags[i]]

    # Update anomaly scatter
    anomaly_scatter.set_offsets(np.column_stack((anomaly_x, anomaly_y)))

    # Update rolling statistics
    if len(data_window) >= ROLLING_WINDOW:
        rolling_mean = pd.Series(y_data).rolling(window=ROLLING_WINDOW).mean()
        rolling_std = pd.Series(y_data).rolling(window=ROLLING_WINDOW).std()

        upper_bound = rolling_mean + (STD_THRESHOLD * rolling_std)
        lower_bound = rolling_mean - (STD_THRESHOLD * rolling_std)

        # Align rolling statistics with current window
        rolling_mean_line.set_data(x_data, rolling_mean)
        upper_bound_line.set_data(x_data, upper_bound)
        lower_bound_line.set_data(x_data, lower_bound)
    else:
        rolling_mean_line.set_data([], [])
        upper_bound_line.set_data([], [])
        lower_bound_line.set_data([], [])

    # Adjust the plot limits
    if len(x_data) > 0:
        ax.set_xlim(max(0, x_data[0]), x_data[-1] + 10)
    if len(y_data) > 0:
        if len(data_window) >= ROLLING_WINDOW:
            ymin = min(min(y_data), min(lower_bound[-len(y_data):])) - 1
            ymax = max(max(y_data), max(upper_bound[-len(y_data):])) + 1
        else:
            ymin = min(y_data) - 1
            ymax = max(y_data) + 1
        ax.set_ylim(ymin, ymax)

    # Update statistics text
    anomaly_rate = (total_anomalies / len(all_y) * 100) if len(all_y) > 0 else 0
    stats_text.set_text(f'Total Points: {len(all_y)}\n'
                        f'Anomalies in Window: {sum(anomalies_window)}\n'
                        f'Total Anomalies: {total_anomalies}\n'
                        f'Anomaly Rate: {anomaly_rate:.2f}%')

    return line, anomaly_scatter, rolling_mean_line, upper_bound_line, lower_bound_line, stats_text

# Callback functions for interactive widgets
def update_threshold(val):
    global STD_THRESHOLD
    STD_THRESHOLD = slider_threshold.val

def reset(event):
    global data_window, anomalies_window, x_window, index, total_anomalies, all_x, all_y, new_points_since_train, model, model_initialized, scaler
    data_window.clear()
    anomalies_window.clear()
    x_window.clear()
    all_x.clear()
    all_y.clear()
    index = 0
    total_anomalies = 0
    new_points_since_train = 0
    line.set_data([], [])
    anomaly_scatter.set_offsets([])
    rolling_mean_line.set_data([], [])
    upper_bound_line.set_data([], [])
    lower_bound_line.set_data([], [])
    stats_text.set_text('')
    model_initialized = False

    # Reset One-Class SVM
    model = OneClassSVM(kernel=SVM_KERNEL, nu=SVM_NU, gamma=SVM_GAMMA)
    scaler = StandardScaler()

    # Clear the CSV log
    with open('anomalies_log_svm.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Value', 'Timestamp', 'Anomaly_Type'])

    print("System has been reset.")

# ----------------------------- Set up the plot -----------------------------
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(14, 7))

# Plot elements
line, = ax.plot([], [], 'b-', label='Normal Data')                # Blue line
anomaly_scatter = ax.scatter([], [], c='red', label='Anomaly')    # Red markers
rolling_mean_line, = ax.plot([], [], 'g--', label='Rolling Mean')  # Green dashed
upper_bound_line, = ax.plot([], [], 'r--', label='Upper Bound')   # Red dashed
lower_bound_line, = ax.plot([], [], 'y--', label='Lower Bound')   # Yellow dashed

# Plot settings
ax.set_title('Real-Time Anomaly Detection with One-Class SVM', fontsize=16)
ax.set_xlabel('Index', fontsize=14)
ax.set_ylabel('Value', fontsize=14)
ax.legend(loc='upper left', fontsize=12)
ax.grid(True)

# Add real-time statistics text box
stats_text = ax.text(0.98, 0.95, '', transform=ax.transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))

# Add Slider for STD_THRESHOLD
ax_slider = plt.axes([0.25, 0.02, 0.50, 0.02])
slider_threshold = Slider(
    ax=ax_slider,
    label='STD Threshold',
    valmin=1,
    valmax=5,
    valinit=STD_THRESHOLD,
    valstep=0.1
)
slider_threshold.on_changed(update_threshold)

# Add Reset Button
ax_button = plt.axes([0.80, 0.02, 0.1, 0.04])
button = Button(ax_button, 'Reset', hovercolor='0.975')
button.on_clicked(reset)

# ----------------------------- Initialize the animation -----------------------------
ani = animation.FuncAnimation(
    fig,
    update_plot,
    fargs=(line, anomaly_scatter, rolling_mean_line, upper_bound_line, lower_bound_line, stats_text),
    interval=UPDATE_INTERVAL,
    blit=False
)

plt.tight_layout()
plt.show()
