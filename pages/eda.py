import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import numpy as np

dash.register_page(__name__)

# --- Helper Function ---
def create_empty_figure(message="Please select options to render the chart."):
    fig = go.Figure()
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      annotations=[{"text": message, "xref": "paper", "yref": "paper", "showarrow": False, "font": {"size": 16, "color": "grey"}}])
    return fig

# --- Layout Definition (Corrected Syntax) ---
layout = dbc.Container([
    dbc.Row(dbc.Col(html.H3("Exploratory Data Analysis (EDA)", className="mt-4 mb-4"), width=12)),
    dbc.Row(dbc.Col(dbc.Alert(id='eda-error-alert', color="danger", is_open=False, duration=8000), width=12)),
    dbc.Row(dbc.Col(dbc.Alert(id='eda-info-alert', color="info", is_open=False), width=12)),

    dbc.Row([
        # --- CONTROL PANEL ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Plotting Controls"),
                dbc.CardBody([
                    dbc.Label("1. Choose Chart Type:"),
                    dcc.Dropdown(id='chart-type-selector', options=[
                        {'label': 'Time Series Plot', 'value': 'timeseries'},
                        {'label': 'Candlestick Chart', 'value': 'candlestick'},
                        {'label': 'Distribution (Histogram)', 'value': 'histogram'},
                        {'label': 'Scatter Plot', 'value': 'scatter'},
                        {'label': 'Correlation Heatmap', 'value': 'heatmap'},
                    ], value='timeseries'),
                    html.Hr(),

                    # --- All controls are now permanently in the layout, hidden/shown as needed ---
                    html.Div(id='timeseries-controls', children=[
                        dbc.Label("Select Date Column:"), dcc.Dropdown(id="ts-date-selector"), html.Br(),
                        dbc.Label("Select Value Column:"), dcc.Dropdown(id="ts-value-selector"),
                    ]),
                    html.Div(id='candlestick-controls', style={'display': 'none'}, children=[
                        dbc.Label("Select Date Column:"), dcc.Dropdown(id="cs-date-selector"), html.Br(),
                        dbc.Label("Map Open Column:"), dcc.Dropdown(id="cs-open-selector"), html.Br(),
                        dbc.Label("Map High Column:"), dcc.Dropdown(id="cs-high-selector"), html.Br(),
                        dbc.Label("Map Low Column:"), dcc.Dropdown(id="cs-low-selector"), html.Br(),
                        dbc.Label("Map Close Column:"), dcc.Dropdown(id="cs-close-selector"),
                    ]),
                    html.Div(id='histogram-controls', style={'display': 'none'}, children=[
                        dbc.Label("Select Column:"), dcc.Dropdown(id="hist-selector"),
                    ]),
                    html.Div(id='scatter-controls', style={'display': 'none'}, children=[
                        dbc.Label("Select X-Axis:"), dcc.Dropdown(id="scatter-x-selector"), html.Br(),
                        dbc.Label("Select Y-Axis:"), dcc.Dropdown(id="scatter-y-selector"),
                    ]),
                    html.Div(id='heatmap-controls', style={'display': 'none'}, children=[
                         html.P("This chart uses all numeric columns to show correlations.", className="text-muted")
                    ])
                ])
            ])
        ], width=3),

        # --- GRAPH DISPLAY ---
        dbc.Col(
            dcc.Loading(
                id="loading-spinner",
                color="primary",
                children=dcc.Graph(id='eda-dynamic-plot', style={'height': '75vh'}, figure=create_empty_figure())
            ),
            width=9
        )
    ]), # <-- THIS IS THE LINE THAT HAD THE SYNTAX ERROR. IT IS NOW CORRECT.

], fluid=True)


# --- UI Controls Callback ---
@callback(
    [Output(id, 'options') for id in ['ts-date-selector', 'ts-value-selector', 'cs-date-selector', 
                                     'cs-open-selector', 'cs-high-selector', 'cs-low-selector', 'cs-close-selector',
                                     'hist-selector', 'scatter-x-selector', 'scatter-y-selector']],
    Input('store-data', 'data')
)
def populate_all_dropdowns(json_data):
    if not json_data: return [[] for _ in range(10)]
    df = pd.read_json(io.StringIO(json_data), orient='split')
    all_cols = [{'label': col, 'value': col} for col in df.columns]
    numeric_cols = [{'label': col, 'value': col} for col in df.select_dtypes(include=np.number).columns]
    return [all_cols, numeric_cols, all_cols, numeric_cols, numeric_cols, numeric_cols, numeric_cols, all_cols, numeric_cols, numeric_cols]

# --- Visibility Callback ---
@callback(
    [Output(f'{chart_type}-controls', 'style') for chart_type in ['timeseries', 'candlestick', 'histogram', 'scatter', 'heatmap']],
    Input('chart-type-selector', 'value')
)
def update_control_visibility(chart_type):
    styles = [{'display': 'none'}] * 5
    type_map = {'timeseries': 0, 'candlestick': 1, 'histogram': 2, 'scatter': 3, 'heatmap': 4}
    if chart_type in type_map:
        styles[type_map[chart_type]] = {'display': 'block'}
    return styles

# --- Main Graphing Callback ---
@callback(
    Output('eda-dynamic-plot', 'figure'),
    Output('eda-error-alert', 'children'), Output('eda-error-alert', 'is_open'),
    Output('eda-info-alert', 'children'), Output('eda-info-alert', 'is_open'),
    [
        Input('chart-type-selector', 'value'),
        Input('ts-date-selector', 'value'), Input('ts-value-selector', 'value'),
        Input('cs-date-selector', 'value'), Input('cs-open-selector', 'value'),
        Input('cs-high-selector', 'value'), Input('cs-low-selector', 'value'), Input('cs-close-selector', 'value'),
        Input('hist-selector', 'value'),
        Input('scatter-x-selector', 'value'), Input('scatter-y-selector', 'value'),
    ],
    State('store-data', 'data')
)
def update_master_graph(chart_type, ts_date, ts_val, cs_date, cs_open, cs_high, cs_low, cs_close, hist_val, sc_x, sc_y, json_data):
    if not json_data: return dash.no_update
    df = pd.read_json(io.StringIO(json_data), orient='split')
    info_message, show_info = "", False
    try:
        if chart_type == 'timeseries':
            if not all([ts_date, ts_val]): return create_empty_figure(), "", False, "", False
            if not pd.api.types.is_numeric_dtype(df[ts_date]):
                df[ts_date] = pd.to_datetime(df[ts_date], errors='coerce')
                if df[ts_date].isnull().any(): raise ValueError(f"Column '{ts_date}' could not be converted to dates.")
            else: raise ValueError(f"Column '{ts_date}' is numeric. Please select a date/time column.")
            df = df.sort_values(by=ts_date)
            fig = px.line(df, x=ts_date, y=ts_val, title=f"Time Series of {ts_val}", template="plotly_white")
        elif chart_type == 'candlestick':
            selections = [cs_date, cs_open, cs_high, cs_low, cs_close]
            if not all(selections): return create_empty_figure("Please map all Date and OHLC columns."), "", False, "", False
            df[cs_date] = pd.to_datetime(df[cs_date], errors='coerce')
            if df[cs_date].isnull().any(): raise ValueError(f"Column '{cs_date}' could not be converted to dates.")
            df = df.sort_values(by=cs_date)
            fig = go.Figure(data=[go.Candlestick(x=df[cs_date], open=df[cs_open], high=df[cs_high], low=df[cs_low], close=df[cs_close], increasing=dict(line=dict(color='#26A69A')), decreasing=dict(line=dict(color='#EF5350')))])
            fig.update_layout(title="Financial Candlestick Chart", xaxis_rangeslider_visible=True, template="plotly_white")
        elif chart_type == 'histogram':
            if not hist_val: return create_empty_figure(), "", False, "", False
            fig = px.histogram(df, x=hist_val, title=f"Distribution of {hist_val}", marginal="box", template="plotly_white")
        elif chart_type == 'scatter':
            if not all([sc_x, sc_y]): return create_empty_figure(), "", False, "", False
            fig = px.scatter(df, x=sc_x, y=sc_y, title=f"Scatter Plot of {sc_x} vs. {sc_y}", trendline="ols", trendline_color_override="red", template="plotly_white")
        elif chart_type == 'heatmap':
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.shape[1] < 2: raise ValueError("Requires at least two numeric columns.")
            pct_change_df = numeric_df.pct_change().dropna()
            fig = px.imshow(pct_change_df.corr(), text_auto=True, title="Correlation Heatmap of Daily Percentage Changes", color_continuous_scale='RdBu_r', aspect="auto")
            info_message, show_info = "This heatmap shows the correlation between the daily percentage changes of your numeric columns.", True
        else: return create_empty_figure(), "", False, "", False
        return fig, "", False, info_message, show_info
    except Exception as e:
        return create_empty_figure(), str(e), True, "", False