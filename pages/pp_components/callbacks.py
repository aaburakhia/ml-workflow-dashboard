# pages/pp_components/callbacks.py

import pandas as pd
import io
import dash
from dash import callback, Input, Output, State, html, dash_table
from logic.preprocessing_jobs import run_preprocessing_job

def register_callbacks(app):
    @app.callback(
        Output("store-processed-data", "data"),
        Output("processed-data-preview", "children"),
        Output("pp-alert", "children"),
        Output("pp-alert", "is_open"),
        Input("run-pp-btn", "n_clicks"),
        [
            State("store-data", "data"),
            State("missing-value-strategy", "value"),
            State("scaling-strategy", "value")
        ],
        prevent_initial_call=True
    )
    def apply_preprocessing(n_clicks, json_raw_data, missing_val_strat, scaling_strat):
        if not json_raw_data:
            # Return 4 values to match the 4 Outputs
            return dash.no_update, "No raw data found. Please upload data on the Home page.", "No raw data found.", True

        try:
            raw_df = pd.read_json(io.StringIO(json_raw_data), orient='split')

            processed_df = run_preprocessing_job(raw_df, missing_val_strat, scaling_strat)

            preview_table = dash_table.DataTable(
                data=processed_df.head(10).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in processed_df.columns],
                style_table={'overflowX': 'auto'}
            )

            processed_json = processed_df.to_json(orient='split')

            return processed_json, preview_table, "Preprocessing applied successfully!", True

        except Exception as e:
            # --- CORRECTED THIS BLOCK ---
            # Ensure we always return 4 values, even on failure.
            error_message = f"An error occurred: {e}"
            return dash.no_update, error_message, error_message, True