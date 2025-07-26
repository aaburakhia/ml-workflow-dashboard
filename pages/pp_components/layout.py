# pages/pp_components/layout.py

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

preprocessing_layout = dbc.Container(
    children=[
        dbc.Row(dbc.Col(html.H1("Data Preprocessing", className="mb-4 mt-3 text-center fw-bold"))),
        dbc.Row(dbc.Col(dbc.Alert(id="pp-alert", is_open=False, dismissable=True, duration=6000))),

        dbc.Row([
            # --- CONTROL PANEL ---
            dbc.Col(xs=12, lg=4, children=[
                dbc.Card([
                    dbc.CardHeader(html.H5("Preprocessing Controls")),
                    dbc.CardBody([
                        dbc.Label("1. Handle Missing Values", className="fw-bold"),
                        dbc.RadioItems(
                            id="missing-value-strategy",
                            options=[
                                {'label': 'Fill with Mean', 'value': 'mean'},
                                {'label': 'Fill with Median', 'value': 'median'},
                                {'label': 'Drop Rows with Missing Values', 'value': 'drop'},
                            ],
                            value='mean',
                            className="mb-3"
                        ),

                        dbc.Label("2. Scale Numerical Features", className="fw-bold"),
                        dbc.RadioItems(
                            id="scaling-strategy",
                            options=[
                                {'label': 'Standard Scaling (Z-score)', 'value': 'standard'},
                                {'label': 'No Scaling', 'value': 'none'},
                            ],
                            value='standard',
                            className="mb-4"
                        ),

                        dbc.Button("Apply Preprocessing", id="run-pp-btn", color="primary", className="w-100"),
                    ])
                ])
            ]),

            # --- DATA PREVIEW ---
            dbc.Col(xs=12, lg=8, children=[
                dbc.Card([
                    dbc.CardHeader(html.H5("Processed Data Preview")),
                    dbc.CardBody(id="processed-data-preview", children=[
                        html.Small("Apply preprocessing to see a preview of the cleaned data.", className="text-muted")
                    ])
                ])
            ])
        ])
    ],
    fluid=True
)