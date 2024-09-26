import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import random
from collections import deque
from matplotlib.widgets import Slider, Button
from sklearn.cluster import DBSCAN
# from playsound import playsound
import threading
import datetime
import csv
import os

# ----------------------------- Parameters -----------------------------
WINDOW_SIZE = 200               # Number of data points to display
ANOMALY_PROBABILITY = 0.02      # Probability of injecting an anomaly
ANOMALY_MULTIPLIER = 5          # Magnitude of injected anomalies
UPDATE_INTERVAL = 100           # Milliseconds between plot updates
DBSCAN_EPS = 0.5                # DBSCAN parameter: maximum distance between two samples
DBSCAN_MIN_SAMPLES = 5          # DBSCAN parameter: minimum number of samples in a neighborhood

# Initialize data storage
all_x = []                      # Stores all x-data points
all_y = []                      # Stores all y-data points
data_window = deque(maxlen=WINDOW_SIZE)      # Sliding window y-data for plotting
x_window = deque(maxlen=WINDOW_SIZE)        # Sliding window x-data for plotting
anomalies_window = deque(maxlen=WINDOW_SIZE) # Sliding window anomaly flags for plotting
index = 0
total_anomalies = 0            # Counter for total anomalies since start

# Ensure anomalies_log_dbscan.csv exists
if not os.path.isfile('anomalies_log_dbscan.csv'):
    with open('anomalies_log_dbscan.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Value', 'Timestamp'])

# Function to generate the next data point
def generate_data_point():
    global index, total_anomalies
    index += 1
    # Generate normal data
    value = np.random.normal(loc=0.0, scale=1.0)

    # Randomly decide whether to inject an anomaly
    if random.random() < ANOMALY_PROBABILITY:
        # Inject an anomaly
        std_dev = np.std(data_window) if len(data_window) > 1 else 1.0
        value += random.choice([-1, 1]) * ANOMALY_MULTIPLIER * std_dev
        is_anomaly = True
        total_anomalies += 1

        # Log anomaly
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('anomalies_log_dbscan.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([index, value, timestamp])

        # Play alert sound in a separate thread
        # play_alert_sound()
    else:
        is_anomaly = False

    return index, value, is_anomaly

# Function to play alert sound
# def play_alert_sound():
#     threading.Thread(target=playsound, args=('alert.mp3',), daemon=True).start()

# Function to perform DBSCAN and update anomalies
def detect_anomalies_dbscan():
    if len(data_window) < DBSCAN_MIN_SAMPLES:
        return [False]*len(data_window)
    
    # Reshape data for DBSCAN (needs 2D array)
    X = np.array(list(data_window)).reshape(-1, 1)
    
    # Initialize and fit DBSCAN
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    labels = db.fit_predict(X)
    
    # In DBSCAN, label -1 is considered as noise
    anomalies = [label == -1 for label in labels]
    return anomalies

# Function to update the plot
def update(frame, line, anomaly_scatter, stats_text):
    global total_anomalies
    # Generate new data point
    idx, val, is_anomaly = generate_data_point()

    # Append to all data lists
    all_x.append(idx)
    all_y.append(val)

    # Append to windowed deques
    x_window.append(idx)
    data_window.append(val)

    # Detect anomalies using DBSCAN
    anomalies = detect_anomalies_dbscan()
    anomalies_window.append(anomalies[-1] if anomalies else False)

    # Check if the latest point is anomaly
    if anomalies_window[-1] and not is_anomaly:
        total_anomalies += 1

        # Log anomaly
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('anomalies_log_dbscan.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([idx, val, timestamp])

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

    # Adjust the plot limits
    if len(x_data) > 0:
        ax.set_xlim(max(0, x_data[0]), x_data[-1] + 10)
    if len(y_data) > 0:
        ymin = min(y_data) - 1
        ymax = max(y_data) + 1
        ax.set_ylim(ymin, ymax)

    # Update statistics text
    stats_text.set_text(f'Total Points: {len(all_y)}\n'
                        f'Anomalies in Window: {sum(anomaly_flags)}\n'
                        f'Total Anomalies: {total_anomalies}\n'
                        f'Anomaly Rate: {(total_anomalies / len(all_y) * 100):.2f}%')

    return line, anomaly_scatter, stats_text

# Callback functions for interactive widgets
def update_eps(val):
    global DBSCAN_EPS
    DBSCAN_EPS = slider_eps.val
    slider_eps_label.set_text(f'DBSCAN eps: {DBSCAN_EPS:.2f}')

def update_min_samples(val):
    global DBSCAN_MIN_SAMPLES
    DBSCAN_MIN_SAMPLES = int(slider_min_samples.val)
    slider_min_samples_label.set_text(f'DBSCAN min_samples: {DBSCAN_MIN_SAMPLES}')

def reset(event):
    global data_window, anomalies_window, x_window, index, total_anomalies, all_x, all_y
    data_window.clear()
    anomalies_window.clear()
    x_window.clear()
    all_x.clear()
    all_y.clear()
    index = 0
    total_anomalies = 0
    line.set_data([], [])
    anomaly_scatter.set_offsets([])
    stats_text.set_text('')
    
    # Clear the CSV log
    with open('anomalies_log_dbscan.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Value', 'Timestamp'])

# ----------------------------- Set up the plot -----------------------------
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(14, 7))
plt.subplots_adjust(bottom=0.25)  # Make space for sliders and buttons

line, = ax.plot([], [], 'b-', label='Normal Data')                # Blue line
anomaly_scatter = ax.scatter([], [], c='red', label='Anomaly')    # Red markers

ax.set_title('Real-Time Anomaly Detection with DBSCAN', fontsize=16)
ax.set_xlabel('Index', fontsize=14)
ax.set_ylabel('Value', fontsize=14)
ax.legend(loc='upper left', fontsize=12)
ax.grid(True)

# Add real-time statistics text box
stats_text = ax.text(0.98, 0.95, '', transform=ax.transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))

# Add Slider for DBSCAN eps
ax_slider_eps = plt.axes([0.25, 0.15, 0.5, 0.03])
slider_eps = Slider(
    ax=ax_slider_eps,
    label='DBSCAN eps',
    valmin=0.1,
    valmax=5.0,
    valinit=DBSCAN_EPS,
    valstep=0.1
)
slider_eps.on_changed(update_eps)

# Add label for DBSCAN eps
slider_eps_label = ax_slider_eps.text(0.5, -0.5, f'DBSCAN eps: {DBSCAN_EPS:.2f}', ha='center', va='top')

# Add Slider for DBSCAN min_samples
ax_slider_min_samples = plt.axes([0.25, 0.1, 0.5, 0.03])
slider_min_samples = Slider(
    ax=ax_slider_min_samples,
    label='DBSCAN min_samples',
    valmin=1,
    valmax=20,
    valinit=DBSCAN_MIN_SAMPLES,
    valstep=1
)
slider_min_samples.on_changed(update_min_samples)

# Add label for DBSCAN min_samples
slider_min_samples_label = ax_slider_min_samples.text(0.5, -0.5, f'DBSCAN min_samples: {DBSCAN_MIN_SAMPLES}', ha='center', va='top')

# Add Reset Button
ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(ax_button, 'Reset', hovercolor='0.975')
button.on_clicked(reset)

# ----------------------------- Initialize the animation -----------------------------
ani = animation.FuncAnimation(
    fig,
    update,
    fargs=(line, anomaly_scatter, stats_text),
    interval=UPDATE_INTERVAL,
    blit=False
)

plt.tight_layout()
plt.show()