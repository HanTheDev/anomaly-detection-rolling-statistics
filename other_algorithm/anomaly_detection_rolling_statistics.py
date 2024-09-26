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

# ----------------------------- Parameters -----------------------------
WINDOW_SIZE = 200               # Increased window size to display more data
ANOMALY_PROBABILITY = 0.02      # Probability of an anomaly
ANOMALY_MULTIPLIER = 5          # How much an anomaly deviates from the normal data
UPDATE_INTERVAL = 100           # Milliseconds between plot updates
ROLLING_WINDOW = 50             # Window size for rolling statistics
STD_THRESHOLD = 3               # Number of standard deviations to consider for anomalies

# Rolling Statistics Visualization Parameters
ROLLING_WINDOW_VISUAL = ROLLING_WINDOW
UPPER_BOUND_MULTIPLE = STD_THRESHOLD
LOWER_BOUND_MULTIPLE = -STD_THRESHOLD

# Initialize data storage
all_x = []                      # Stores all x-data points
all_y = []                      # Stores all y-data points
data_window = deque(maxlen=WINDOW_SIZE)      # Windowed y-data for plotting
anomalies_window = deque(maxlen=WINDOW_SIZE) # Windowed anomaly flags for plotting
x_window = deque(maxlen=WINDOW_SIZE)        # Windowed x-data for plotting
index = 0
total_anomalies = 0            # Counter for total anomalies since start

# Ensure anomalies_log.csv exists
if not os.path.isfile('anomalies_log_rolling_stats.csv'):
    with open('anomalies_log_rolling_stats.csv', 'w', newline='') as csvfile:
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
        with open('anomalies_log_rolling_stats.csv', 'a', newline='') as csvfile:
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

# Function to update the plot
def update(frame, line, anomaly_scatter, rolling_mean_line, upper_bound_line, lower_bound_line, stats_text):
    global total_anomalies
    # Generate new data point
    idx, val, is_anomaly = generate_data_point()

    # Append to all data lists
    all_x.append(idx)
    all_y.append(val)

    # Append to windowed deques
    x_window.append(idx)
    data_window.append(val)
    anomalies_window.append(is_anomaly)

    # Anomaly detection using rolling statistics
    detected = False
    if len(data_window) >= ROLLING_WINDOW:
        series = pd.Series(list(data_window))
        rolling_mean = series.rolling(window=ROLLING_WINDOW).mean().iloc[-1]
        rolling_std = series.rolling(window=ROLLING_WINDOW).std().iloc[-1]
        if not pd.isna(rolling_mean) and not pd.isna(rolling_std):
            if abs(val - rolling_mean) > STD_THRESHOLD * rolling_std:
                detected = True

    # Update anomalies_window if anomaly is detected by detection logic
    if detected and not is_anomaly:
        anomalies_window.pop()       # Remove the last flag added by generate_data_point
        anomalies_window.append(True)
        total_anomalies += 1

        # Log anomaly
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('anomalies_log_rolling_stats.csv', 'a', newline='') as csvfile:
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

    # Update rolling statistics
    if len(data_window) >= ROLLING_WINDOW_VISUAL:
        series_full = pd.Series(list(data_window))
        rolling_mean_values = series_full.rolling(window=ROLLING_WINDOW_VISUAL).mean().tolist()
        rolling_std_values = series_full.rolling(window=ROLLING_WINDOW_VISUAL).std().tolist()

        upper_bound = [m + (UPPER_BOUND_MULTIPLE * s) for m, s in zip(rolling_mean_values, rolling_std_values)]
        lower_bound = [m - (UPPER_BOUND_MULTIPLE * s) for m, s in zip(rolling_mean_values, rolling_std_values)]

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

# Callback functions for interactive widgets
def update_threshold(val):
    global STD_THRESHOLD
    STD_THRESHOLD = slider_threshold.val

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
    rolling_mean_line.set_data([], [])
    upper_bound_line.set_data([], [])
    lower_bound_line.set_data([], [])
    stats_text.set_text('')

    # Clear the CSV log
    with open('anomalies_log_rolling_stats.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Value', 'Timestamp'])

# ----------------------------- Set up the plot -----------------------------
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(14, 7))
line, = ax.plot([], [], 'b-', label='Normal Data')                # Blue line
anomaly_scatter = ax.scatter([], [], c='red', label='Anomaly')    # Red markers
rolling_mean_line, = ax.plot([], [], 'g--', label='Rolling Mean')  # Green dashed
upper_bound_line, = ax.plot([], [], 'r--', label='Upper Bound')   # Red dashed
lower_bound_line, = ax.plot([], [], 'y--', label='Lower Bound')   # Yellow dashed

ax.set_title('Real-Time Anomaly Detection with Enhanced Visualization', fontsize=16)
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
    update,
    fargs=(line, anomaly_scatter, rolling_mean_line, upper_bound_line, lower_bound_line, stats_text),
    interval=UPDATE_INTERVAL,
    blit=False
)

plt.tight_layout()
plt.show()
