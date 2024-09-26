# File: app.py

import dash
from dash.dependencies import Output, Input, State
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import random
from collections import deque
import datetime
import threading
import os
import csv

# ----------------------------- Parameters -----------------------------
WINDOW_SIZE = 200               # Number of data points to display on the graph
ANOMALY_PROBABILITY = 0.02      # Probability of injecting an anomaly
ANOMALY_MULTIPLIER = 5          # Multiplier for anomaly deviation
UPDATE_INTERVAL = 1000          # Update interval in milliseconds
ROLLING_WINDOW = 50             # Window size for rolling statistics
STD_THRESHOLD_DEFAULT = 3       # Default number of standard deviations for anomaly detection

# ----------------------------- Initialize Data Structures -----------------------------
all_x = []                       # Stores all x-data points
all_y = []                       # Stores all y-data points
data_window = deque(maxlen=WINDOW_SIZE)       # Windowed y-data for plotting
anomalies_window = deque(maxlen=WINDOW_SIZE)  # Windowed anomaly flags for plotting
x_window = deque(maxlen=WINDOW_SIZE)         # Windowed x-data for plotting
index = 0
total_anomalies = 0             # Counter for total anomalies since start

# Ensure anomalies_log.csv exists
if not os.path.isfile('anomalies_log.csv'):
    with open('anomalies_log.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Value', 'Timestamp'])

# ----------------------------- Dash App Initialization -----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Real-Time Anomaly Detection Dashboard"

# ----------------------------- Helper Functions -----------------------------
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
        with open('anomalies_log.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([index, value, timestamp])

        # Play alert sound (Note: Implemented differently in web apps)
        # play_alert_sound()  # We'll discuss this later
    else:
        is_anomaly = False

    return index, value, is_anomaly

# ----------------------------- Layout Definition -----------------------------
import dash_bootstrap_components as dbc

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Real-Time Anomaly Detection Dashboard"), className="mb-2")
    ]),
    dbc.Row([
        dbc.Col(html.Div([
            dcc.Graph(id='live-graph', animate=False),
            dcc.Interval(
                id='graph-update',
                interval=UPDATE_INTERVAL,  # in milliseconds
                n_intervals=0
            )
        ]))
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Standard Deviation Threshold for Anomaly Detection"),
            dcc.Slider(
                id='threshold-slider',
                min=1,
                max=5,
                step=0.1,
                value=STD_THRESHOLD_DEFAULT,
                marks={i: f"{i}" for i in range(1, 6)}
            )
        ], width=8),
        dbc.Col([
            html.Button('Reset Data', id='reset-button', n_clicks=0, className='btn btn-primary')
        ], width=4)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='statistics-display'), className="mt-4")
    ])
], fluid=True)

# ----------------------------- Callback Definitions -----------------------------

@app.callback(
    [Output('live-graph', 'figure'),
     Output('statistics-display', 'children')],
    [Input('graph-update', 'n_intervals')],
    [State('threshold-slider', 'value')]
)
def update_graph_scatter(n, std_threshold):
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
            if abs(val - rolling_mean) > std_threshold * rolling_std:
                detected = True

    # Update anomalies_window if anomaly is detected by detection logic
    if detected and not is_anomaly:
        anomalies_window.pop()       # Remove the last flag added by generate_data_point
        anomalies_window.append(True)
        total_anomalies += 1

        # Log anomaly
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('anomalies_log.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([idx, val, timestamp])

        # Play alert sound here if possible (limited in Dash)
        # To implement browser-based sound, additional client-side code is required

    # Prepare data for plotting
    y_data = list(data_window)
    x_data = list(x_window)
    anomaly_flags = list(anomalies_window)

    # Create main trace for normal data
    main_trace = go.Scatter(
        x=x_data,
        y=y_data,
        mode='lines',
        name='Normal Data',
        line=dict(color='blue')
    )

    # Create scatter trace for anomalies
    anomaly_x = [x_data[i] for i in range(len(y_data)) if anomaly_flags[i]]
    anomaly_y = [y_data[i] for i in range(len(y_data)) if anomaly_flags[i]]

    anomaly_trace = go.Scatter(
        x=anomaly_x,
        y=anomaly_y,
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=10, symbol='circle')
    )

    # Rolling statistics
    if len(data_window) >= ROLLING_WINDOW:
        series_full = pd.Series(list(data_window))
        rolling_mean_values = series_full.rolling(window=ROLLING_WINDOW).mean().tolist()
        rolling_std_values = series_full.rolling(window=ROLLING_WINDOW).std().tolist()

        upper_bound = [m + (std_threshold * s) for m, s in zip(rolling_mean_values, rolling_std_values)]
        lower_bound = [m - (std_threshold * s) for m, s in zip(rolling_mean_values, rolling_std_values)]

        rolling_mean_trace = go.Scatter(
            x=x_data,
            y=rolling_mean_values[-len(x_data):],
            mode='lines',
            name='Rolling Mean',
            line=dict(color='green', dash='dash')
        )

        upper_bound_trace = go.Scatter(
            x=x_data,
            y=upper_bound[-len(x_data):],
            mode='lines',
            name='Upper Bound',
            line=dict(color='red', dash='dash')
        )

        lower_bound_trace = go.Scatter(
            x=x_data,
            y=lower_bound[-len(x_data):],
            mode='lines',
            name='Lower Bound',
            line=dict(color='red', dash='dash')
        )

        data_traces = [main_trace, rolling_mean_trace, upper_bound_trace, lower_bound_trace, anomaly_trace]
    else:
        data_traces = [main_trace, anomaly_trace]

    # Define the layout
    layout = go.Layout(
        title='Real-Time Anomaly Detection',
        xaxis=dict(range=[min(x_data) if x_data else 0, max(x_data) + 10 if x_data else 100]),
        yaxis=dict(range=[
            min(y_data + [0]) - 1,
            max(y_data + [0]) + 1
        ]) if y_data else dict(range=[-1, 1]),
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(x=0, y=1.2, orientation='h'),
        hovermode='closest'
    )

    figure = go.Figure(data=data_traces, layout=layout)

    # Prepare statistics display
    stats = html.Div([
        html.H5("Statistics"),
        html.P(f"Total Points: {len(all_y)}"),
        html.P(f"Anomalies in Window: {sum(anomalies_window)}"),
        html.P(f"Total Anomalies: {total_anomalies}"),
        html.P(f"Anomaly Rate: {(total_anomalies / len(all_y) * 100) if len(all_y) > 0 else 0:.2f}%")
    ], style={
        'border': '1px solid #d3d3d3',
        'padding': '10px',
        'border-radius': '5px',
        'background-color': '#f9f9f9'
    })

    return figure, stats

# Callback for Reset Button
@app.callback(
    [Output('live-graph', 'figure'),
     Output('statistics-display', 'children')],
    [Input('reset-button', 'n_clicks')],
    prevent_initial_call=True
)
def reset_data(n_clicks):
    global all_x, all_y, data_window, anomalies_window, x_window, index, total_anomalies

    # Clear all data structures
    all_x = []
    all_y = []
    data_window.clear()
    anomalies_window.clear()
    x_window.clear()
    index = 0
    total_anomalies = 0

    # Clear the CSV log
    with open('anomalies_log.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Value', 'Timestamp'])

    # Initialize an empty figure
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title='Real-Time Anomaly Detection',
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[-1, 1]),
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(x=0, y=1.2, orientation='h'),
        hovermode='closest'
    )

    # Initialize statistics display
    stats = html.Div([
        html.H5("Statistics"),
        html.P(f"Total Points: 0"),
        html.P(f"Anomalies in Window: 0"),
        html.P(f"Total Anomalies: 0"),
        html.P(f"Anomaly Rate: 0.00%")
    ], style={
        'border': '1px solid #d3d3d3',
        'padding': '10px',
        'border-radius': '5px',
        'background-color': '#f9f9f9'
    })

    return empty_fig, stats

# ----------------------------- Run the App -----------------------------
if __name__ == '__main__':
    app.run_server(debug=True)


server = app.server