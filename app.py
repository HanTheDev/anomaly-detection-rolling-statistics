# File: app.py

import dash
from dash.dependencies import Output, Input, State
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import random
from collections import deque
import datetime
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
all_x = []                       # Stores all x-data points for cumulative count
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
server = app.server  # Expose the server variable for deployments

# ----------------------------- Helper Functions -----------------------------
def generate_data_point():
    """
    Generates a new data point, randomly injecting an anomaly based on ANOMALY_PROBABILITY.

    Returns:
        tuple: (index, value, is_anomaly)
            - index (int): The sequential index of the data point.
            - value (float): The value of the data point.
            - is_anomaly (bool): Flag indicating whether the data point is an injected anomaly.
    """
    global index, total_anomalies
    index += 1
    # Generate normal data
    value = np.random.normal(loc=0.0, scale=1.0)

    # Randomly decide whether to inject an anomaly
    if random.random() < ANOMALY_PROBABILITY:
        # Calculate the standard deviation of the current data window
        std_dev = np.std(data_window) if len(data_window) > 1 else 1.0

        # Inject an anomaly by adding or subtracting a multiple of the standard deviation
        value += random.choice([-1, 1]) * ANOMALY_MULTIPLIER * std_dev
        is_anomaly = True
        total_anomalies += 1

        # Log anomaly
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open('anomalies_log.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([index, value, timestamp])

        # Note: Sound alerts handled via clientside callbacks
    else:
        is_anomaly = False

    return index, value, is_anomaly

# ----------------------------- Layout Definition -----------------------------
# Define the layout of the Dash app using Bootstrap components for styling
app.layout = dbc.Container([
    # Header Row: Displays the dashboard titl
    dbc.Row([
        dbc.Col(html.H1("Real-Time Anomaly Detection Dashboard"), className="mb-2")
    ]),
    # Graph
    dbc.Row([
        dbc.Col(html.Div([
            # Live Graph: Displays the real-time data and anomalies
            dcc.Graph(id='live-graph', animate=False),
            # Interval Component: Triggers periodic updates of the graph
            dcc.Interval(
                id='graph-update',
                interval=UPDATE_INTERVAL,  # in milliseconds
                n_intervals=0 # Number of times the interval has passed
            ),
            # Store Component: Holds a flag to trigger sound alerts
            dcc.Store(id='anomaly-store', data=False),
        ]))
    ]),

    # Controls Row: Contains the threshold slider and reset button
    dbc.Row([
        # Slider for adjusting the standard deviation threshold
        dbc.Col([
            html.Label("Standard Deviation Threshold for Anomaly Detection"),
            dcc.Slider(
                id='threshold-slider',
                min=1,                   # Minimum threshold value
                max=5,                   # Maximum threshold value
                step=0.1,                # Increment step
                value=STD_THRESHOLD_DEFAULT,  # Default value
                marks={i: f"{i}" for i in range(1, 6)},  # Label marks on the slider
                tooltip={"placement": "bottom", "always_visible": True},  # Tooltip display
            )
        ], width=8),  # Occupies 8 out of 12 Bootstrap columns
        # Reset Button: Clears all data and resets the dashboard
        dbc.Col([
            html.Button('Reset Data', id='reset-button', n_clicks=0, className='btn btn-primary')
        ], width=4)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='statistics-display'), className="mt-4")
    ])
], fluid=True) # 'fluid=True' makes the container responsive

# ----------------------------- Callback Definitions -----------------------------

@app.callback(
    [
        Output('live-graph', 'figure'),            # Updates the live graph
        Output('statistics-display', 'children'),   # Updates the statistics display
        Output('anomaly-store', 'data')            # Triggers audio alert if True
    ],
    [
        Input('graph-update', 'n_intervals'),      # Triggered by the Interval component
        Input('reset-button', 'n_clicks')          # Triggered by the Reset button
    ],
    [
        State('threshold-slider', 'value')          # Current value of the threshold slider
    ]
)
def update_graph(n, reset_clicks, std_threshold):
    """
    Callback function to update the live graph, statistics, and trigger sound alerts.

    This function handles two main actions:
    1. Generating new data points and updating the graph.
    2. Resetting the data when the reset button is clicked.

    Args:
        n (int): Number of intervals that have passed (unused).
        reset_clicks (int): Number of times the reset button has been clicked.
        std_threshold (float): Current standard deviation threshold from the slider.

    Returns:
        tuple: (figure, stats, is_new_anomaly)
            - figure (dict): The updated Plotly figure.
            - stats (html.Div): The updated statistics display.
            - is_new_anomaly (bool): Flag indicating if a new anomaly was detected.
    """
    global all_x, all_y, data_window, anomalies_window, x_window, index, total_anomalies

    # Determine which input triggered the callback using callback context
    ctx = dash.callback_context

    if not ctx.triggered:
        trigger_id = 'No clicks yet'
    else:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'reset-button':
        # Reset all data structures
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

        # Create an empty figure
        fig = go.Figure()
        fig.update_layout(
            title='Real-Time Anomaly Detection',
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[-1, 1]),
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(x=0, y=1.2, orientation='h'),
            hovermode='closest'
        )

        # Initialize statistics display
        stats = dbc.Card([
            dbc.CardHeader("Statistics"),
            dbc.CardBody([
                html.P(f"Total Points: 0"),
                html.P(f"Anomalies in Window: 0"),
                html.P(f"Total Anomalies: 0"),
                html.P(f"Anomaly Rate: 0.00%")
            ])
        ], style={'width': '18rem'})

        # Return the empty figure, stats, and reset the anomaly-store
        return fig, stats, False

    else:
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

        # Initialize a flag to indicate if a new anomaly was detected via detection logic
        detected = False
        
        # Perform anomaly detection using rolling statistics if sufficient data is available
        if len(data_window) >= ROLLING_WINDOW:
            series = pd.Series(list(data_window))
            rolling_mean = series.rolling(window=ROLLING_WINDOW).mean().iloc[-1]
            rolling_std = series.rolling(window=ROLLING_WINDOW).std().iloc[-1]

            # Check if the current value deviates from the rolling mean by the threshold
            if not pd.isna(rolling_mean) and not pd.isna(rolling_std):
                if abs(val - rolling_mean) > std_threshold * rolling_std:
                    detected = True

        # Update anomalies_window if anomaly is detected by detection logic
        if detected and not is_anomaly:
            anomalies_window.pop()       # Remove the last flag added by generate_data_point
            anomalies_window.append(True) # Append True for the detected anomaly
            total_anomalies += 1          # Increment the cumulative anomalies count

            # Log anomaly
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open('anomalies_log.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([idx, val, timestamp])

            # Trigger audio alert
            is_new_anomaly = True
        else:
            is_new_anomaly = False

        # Prepare data for plotting by converting deques to lists
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
        # If sufficient data is available, calculate and plot rolling statistics
        if len(data_window) >= ROLLING_WINDOW:
            series_full = pd.Series(list(data_window))
            rolling_mean_values = series_full.rolling(window=ROLLING_WINDOW).mean().tolist()
            rolling_std_values = series_full.rolling(window=ROLLING_WINDOW).std().tolist()

            # Calculate upper and lower bounds based on the standard deviation threshold
            upper_bound = [m + (std_threshold * s) for m, s in zip(rolling_mean_values, rolling_std_values)]
            lower_bound = [m - (std_threshold * s) for m, s in zip(rolling_mean_values, rolling_std_values)]

            # Create a trace for the rolling mean as a green dashed line
            rolling_mean_trace = go.Scatter(
                x=x_data,
                y=rolling_mean_values[-len(x_data):],
                mode='lines',
                name='Rolling Mean',
                line=dict(color='green', dash='dash')
            )

            # Create a trace for the upper bound as a red dashed line
            upper_bound_trace = go.Scatter(
                x=x_data,
                y=upper_bound[-len(x_data):],
                mode='lines',
                name='Upper Bound',
                line=dict(color='red', dash='dash')
            )

            # Create a trace for the lower bound as a black dashed line
            lower_bound_trace = go.Scatter(
                x=x_data,
                y=lower_bound[-len(x_data):],
                mode='lines',
                name='Lower Bound',
                line=dict(color='black', dash='dash')
            )

            data_traces = [main_trace, rolling_mean_trace, upper_bound_trace, lower_bound_trace, anomaly_trace]
        else:
            data_traces = [main_trace, anomaly_trace]

        # Define the layout
        fig = go.Figure(data=data_traces)
        fig.update_layout(
            title='Real-Time Anomaly Detection',
            xaxis=dict(range=[min(x_data)-10 if x_data else 0, max(x_data) + 10 if x_data else 100]),
            yaxis=dict(
                range=[
                    min(y_data + [0]) - 1, # Dynamic y-axis lower bound with padding
                    max(y_data + [0]) + 1  # Dynamic y-axis upper bound with padding
                ]
            ) if y_data else dict(range=[-1, 1]),  # Default y-axis range if no data
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(x=0, y=1.2, orientation='h'), # Position the legend above the graph
            hovermode='closest'                       # Display hover information for the closest data point
        )

        # Prepare statistics display
        stats = dbc.Card([
            dbc.CardHeader("Statistics"),
            dbc.CardBody([
                html.P(f"Total Points: {len(all_y)}"),
                html.P(f"Anomalies in Window: {sum(anomalies_window)}"),
                html.P(f"Total Anomalies: {total_anomalies}"),
                html.P(f"Anomaly Rate: {((total_anomalies / len(all_y)) * 100) if len(all_y) > 0 else 0:.2f}%")
            ])
        ], style={'width': '18rem'})

        # Update the anomaly-store data to trigger audio alert if needed
        return fig, stats, is_new_anomaly
# ----------------------------- Run the App -----------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
