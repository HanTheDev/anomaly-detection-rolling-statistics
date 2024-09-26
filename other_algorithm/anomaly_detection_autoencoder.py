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
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# ----------------------------- Parameters -----------------------------
WINDOW_SIZE = 200               # Number of data points to display
ANOMALY_PROBABILITY = 0.02      # Probability of injecting an anomaly
ANOMALY_MULTIPLIER = 5          # Magnitude of anomaly
UPDATE_INTERVAL = 100           # Milliseconds between plot updates
ROLLING_WINDOW = 50             # Window size for rolling statistics
STD_THRESHOLD = 3               # Number of standard deviations for threshold
INITIAL_TRAINING_SIZE = 500     # Number of initial data points for training

# Rolling Statistics Visualization Parameters
ROLLING_WINDOW_VISUAL = ROLLING_WINDOW
UPPER_BOUND_MULTIPLE = 3
LOWER_BOUND_MULTIPLE = -3

# ----------------------------- Initialize Data Storage -----------------------------
all_x = []                      # All x-data points
all_y = []                      # All y-data points
data_window = deque(maxlen=WINDOW_SIZE)       # Windowed y-data for plotting
anomalies_window = deque(maxlen=WINDOW_SIZE)  # Windowed anomaly flags for plotting
x_window = deque(maxlen=WINDOW_SIZE)         # Windowed x-data for plotting
index = 0
total_anomalies = 0            # Total anomalies since start

# ----------------------------- Ensure anomalies_log_autoencoder.csv Exists -----------------------------
if not os.path.isfile('anomalies_log_autoencoder.csv'):
    with open('anomalies_log_autoencoder.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Value', 'Reconstruction_Error', 'Timestamp'])

# ----------------------------- Define Autoencoder Model -----------------------------
def build_autoencoder(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(input_dim, activation='linear')  # Output layer
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ----------------------------- Initialize and Train Autoencoder -----------------------------
autoencoder = build_autoencoder(ROLLING_WINDOW)

# Temporarily collect data for initial training
initial_training_data = []

print("Collecting initial training data...")

while len(initial_training_data) < INITIAL_TRAINING_SIZE:
    # Generate normal data
    value = np.random.normal(loc=0.0, scale=1.0)
    initial_training_data.append(value)

# Prepare training data: sliding window
X_train = []
for i in range(len(initial_training_data) - ROLLING_WINDOW):
    window = initial_training_data[i:i+ROLLING_WINDOW]
    X_train.append(window)

X_train = np.array(X_train)

print("Training Autoencoder...")
autoencoder.fit(X_train, X_train, epochs=20, batch_size=32, verbose=0)
print("Training completed.")

# Determine reconstruction error threshold
X_train_reconstructions = autoencoder.predict(X_train)
mse = np.mean(np.power(X_train - X_train_reconstructions, 2), axis=1)
threshold = np.mean(mse) + 3 * np.std(mse)  # Set threshold (can be adjusted)
print(f"Reconstruction error threshold set to: {threshold:.4f}")

# ----------------------------- Function to Generate Data Point -----------------------------
def generate_data_point():
    global index, total_anomalies
    index += 1
    # Generate normal data
    value = np.random.normal(loc=0.0, scale=1.0)

    # Randomly decide whether to inject an anomaly
    if random.random() < ANOMALY_PROBABILITY:
        # Inject an anomaly
        std_dev = np.std(list(data_window)) if len(data_window) > 1 else 1.0
        value += random.choice([-1, 1]) * ANOMALY_MULTIPLIER * std_dev
        is_anomaly = True
        total_anomalies += 1

        # Log anomaly
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('anomalies_log_autoencoder.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([index, value, 'Injected', timestamp])

        # Play alert sound in a separate thread
        # play_alert_sound()
    else:
        is_anomaly = False

    return index, value, is_anomaly

# ----------------------------- Function to Play Alert Sound -----------------------------
# def play_alert_sound():
#     threading.Thread(target=playsound, args=('alert.mp3',), daemon=True).start()

# ----------------------------- Function to Update the Plot -----------------------------
def update(frame, line, anomaly_scatter, rolling_mean_line, upper_bound_line, lower_bound_line, stats_text):
    global total_anomalies, threshold

    # Generate new data point
    idx, val, is_anomaly_injected = generate_data_point()

    # Append to all data lists
    all_x.append(idx)
    all_y.append(val)

    # Append to windowed deques
    x_window.append(idx)
    data_window.append(val)
    anomalies_window.append(False)  # Default to False, will update if anomaly detected

    # Prepare input for Autoencoder (window of ROLLING_WINDOW)
    if len(data_window) >= ROLLING_WINDOW:
        window_data = np.array(list(data_window))[-ROLLING_WINDOW:]
        window_data = window_data.reshape(1, -1)  # Reshape for model

        # Get reconstruction
        reconstructed = autoencoder.predict(window_data)
        reconstruction_error = np.mean(np.power(window_data - reconstructed, 2))

        # Anomaly detection based on reconstruction error
        is_anomaly_detected = reconstruction_error > threshold

        if is_anomaly_detected:
            anomalies_window.pop()       # Remove last False
            anomalies_window.append(True)
            total_anomalies += 1

            # Log anomaly
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open('anomalies_log_autoencoder.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([idx, val, f'{reconstruction_error:.4f}', timestamp])

            # Play alert sound
            # play_alert_sound()

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
    if len(data_window) >= ROLLING_WINDOW_VISUAL:
        series_full = pd.Series(list(data_window))
        rolling_mean_values = series_full.rolling(window=ROLLING_WINDOW_VISUAL).mean()
        rolling_std_values = series_full.rolling(window=ROLLING_WINDOW_VISUAL).std()

        upper_bound = rolling_mean_values + (UPPER_BOUND_MULTIPLE * rolling_std_values)
        lower_bound = rolling_mean_values - (LOWER_BOUND_MULTIPLE * rolling_std_values)

        # Align the rolling statistics with the windowed data
        rolling_mean_line.set_data(x_data, rolling_mean_values[-len(x_data):])
        upper_bound_line.set_data(x_data, upper_bound[-len(x_data):])
        lower_bound_line.set_data(x_data, lower_bound[-len(x_data):])
    else:
        rolling_mean_line.set_data([], [])
        upper_bound_line.set_data([], [])
        lower_bound_line.set_data([], [])

    # Adjust the plot limits
    if len(x_data) > 0:
        ax.set_xlim(max(0, x_data[0]), x_data[-1] + 10)
    if len(y_data) > 0:
        ymin = min(min(y_data), min(lower_bound[-len(y_data):] if len(data_window) >= ROLLING_WINDOW_VISUAL else [0])) - 1
        ymax = max(max(y_data), max(upper_bound[-len(y_data):] if len(data_window) >= ROLLING_WINDOW_VISUAL else [0])) + 1
        ax.set_ylim(ymin, ymax)

    # Update statistics text
    stats_text.set_text(f'Total Points: {len(all_y)}\n'
                        f'Anomalies in Window: {sum(anomalies_window)}\n'
                        f'Total Anomalies: {total_anomalies}\n'
                        f'Anomaly Rate: {(total_anomalies / len(all_y) * 100):.2f}%')

    return line, anomaly_scatter, rolling_mean_line, upper_bound_line, lower_bound_line, stats_text

# ----------------------------- Callback Functions for Interactive Widgets -----------------------------
def update_threshold(val):
    global STD_THRESHOLD, threshold
    STD_THRESHOLD = slider_threshold.val
    threshold = autoencoder_loss_threshold(STD_THRESHOLD, mse)
    print(f"Updated reconstruction error threshold to: {threshold:.4f}")

def autoencoder_loss_threshold(std_mult, mse_values):
    # Recompute threshold based on new standard deviation multiplier
    return np.mean(mse_values) + std_mult * np.std(mse_values)

def reset(event):
    global data_window, anomalies_window, x_window, index, total_anomalies, all_x, all_y, threshold

    data_window.clear()
    anomalies_window.clear()
    x_window.clear()
    all_x.clear()
    all_y.clear()
    index = 0
    total_anomalies = 0
    line.set_data([], [])
    anomaly_scatter.set_offsets([])
    rolling_mean_line.set_data([], [])
    upper_bound_line.set_data([], [])
    lower_bound_line.set_data([], [])
    stats_text.set_text('')

    # Clear the CSV log
    with open('anomalies_log_autoencoder.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Value', 'Reconstruction_Error', 'Timestamp'])

    print("Data and logs have been reset.")

# ----------------------------- Set up the Plot -----------------------------
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(14, 7))
line, = ax.plot([], [], 'b-', label='Normal Data')                # Blue line
anomaly_scatter = ax.scatter([], [], c='red', label='Anomaly')    # Red markers
rolling_mean_line, = ax.plot([], [], 'g--', label='Rolling Mean')  # Green dashed
upper_bound_line, = ax.plot([], [], 'r--', label='Upper Bound')   # Red dashed
lower_bound_line, = ax.plot([], [], 'y--', label='Lower Bound')   # Red dashed

ax.set_title('Real-Time Anomaly Detection with Autoencoder', fontsize=16)
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

# ----------------------------- Initialize the Animation -----------------------------
ani = animation.FuncAnimation(
    fig,
    update,
    fargs=(line, anomaly_scatter, rolling_mean_line, upper_bound_line, lower_bound_line, stats_text),
    interval=UPDATE_INTERVAL,
    blit=False
)

plt.tight_layout()
plt.show()
