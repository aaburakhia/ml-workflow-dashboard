# pages/me_components/layout.py

from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

def empty_fig(msg="Nothing yet."):
    return go.Figure().update_layout(
        xaxis_visible=False, yaxis_visible=False, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", annotations=[dict(text=msg, xref="paper", yref="paper", showarrow=False, font=dict(size=18, color="#aaa"))]
    )

evaluation_layout = dbc.Container(
    children=[
        dcc.Interval(id='poll-interval', interval=1*1000, n_intervals=0, disabled=True),
        dcc.Store(id="exp-id"),
        dcc.Store(id="model-eval-data-source"),
        dbc.Row(dbc.Col(html.H1("Model Evaluation & MLOps", className="mb-4 mt-3 text-center fw-bold"))),
        dbc.Row(dbc.Col(dbc.Alert(id="eval-alert", is_open=False, dismissable=True, duration=8000))),
        dbc.Row([
            dbc.Col(xs=12, lg=3, children=[
                dbc.Card([
                    dbc.CardHeader(html.H5("1. Data & Task")),
                    dbc.CardBody([
                        dbc.Label("Task Type"),
                        dbc.RadioItems(id="task-type", options=[{"label": "Regression", "value": "regression"}, {"label": "Classification", "value": "classification"}], value="regression", inline=True),
                        html.Hr(),
                        dbc.Label("Target (y)"),
                        dcc.Dropdown(id="target-col", placeholder="Select..."),
                        dbc.Label("Features (X)"),
                        dcc.Dropdown(id="feature-cols", multi=True, placeholder="Select..."),
                        html.Br(),
                        dbc.Label("Train/Test Split (%)"),
                        dcc.Slider(id="split-slider", min=50, max=90, step=5, value=70, marks={i: str(i) for i in range(50, 91, 5)}, tooltip={"placement": "bottom", "always_visible": True}),
                    ])
                ]),
                dbc.Card([
                    dbc.CardHeader(html.H5("2. Model")),
                    dbc.CardBody([
                        dbc.Label("Algorithm"),
                        dcc.Dropdown(id="algo", placeholder="Choose..."),
                        dbc.Label("Trials (AutoML)"),
                        dcc.Input(id="trials", type="number", value=5, min=1, max=50, step=1),
                        html.Br(), html.Br(),
                        dbc.Button("Launch Experiment", id="run-btn", color="primary", className="w-100"),
                    ])
                ], className="mt-4"),
                dbc.Card([
                    dbc.CardHeader(html.H5("3. What-if Simulator")),
                    dbc.CardBody(id="sim-panel", children=[
                        html.Small("Train a model first to enable simulator.", className="text-muted"),
                        html.Div(id="sim-pred", className="mt-3")
                    ])
                ], className="mt-4")
            ]),
            dbc.Col(xs=12, lg=9, children=[
                # This Div is now the master container for our results.
                # It will hold either the progress bar or the final tabs.
                html.Div(id="results-content", children=[
                    # It starts empty, to be populated by the poll callback
                ])
            ])
        ]),
    ],
    fluid=True
)