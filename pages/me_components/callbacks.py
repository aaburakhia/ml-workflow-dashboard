# pages/me_components/callbacks.py

import uuid, json, io, pathlib, threading
import dash
from dash import html, dcc, callback, Input, Output, State, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score, mean_squared_error, precision_score, recall_score, f1_score
import mlflow.sklearn
import joblib
import matplotlib

from logic.ml_jobs import run_automl_job
from .layout import empty_fig

# --- Global State and Setup ---
ROOT, TMP_DIR, client = None, None, None
setup_done = False

def ensure_setup():
    """Ensures that directories and the MLflow client are initialized. Runs only once."""
    global ROOT, TMP_DIR, client, setup_done
    if not setup_done:
        matplotlib.use("Agg")
        ROOT = pathlib.Path(__file__).parent.parent.parent
        mlruns_dir = ROOT / "mlruns"
        mlruns_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file:{mlruns_dir}")
        TMP_DIR = ROOT / "tmp"
        TMP_DIR.mkdir(exist_ok=True)
        setup_done = True

# --- Callback Registration ---
def register_callbacks(app):

    # --- NEW "GATEKEEPER" CALLBACK (Restored) ---
    @app.callback(
        Output("model-eval-data-source", "data"),
        Input("store-data", "data"),
        Input("store-processed-data", "data")
    )
    def select_data_source(raw_json, processed_json):
        """
        This critical callback decides which data to use for this page:
        processed data if it exists, otherwise the original raw data.
        """
        return processed_json if processed_json else raw_json

    @app.callback(
        Output("target-col", "options"), Output("feature-cols", "options"), Output("algo", "options"),
        Input("model-eval-data-source", "data"), # Now correctly listens to the gatekeeper
        Input("task-type", "value")
    )
    def populate(json_data, task):
        if not json_data: return [], [], []
        df = pd.read_json(io.StringIO(json_data), orient="split")
        if task == "regression":
            targets = list(df.select_dtypes(include=np.number).columns)
            algos = [{"label": "Random Forest Reg", "value": "rf_reg"}, {"label": "Linear Reg", "value": "lin_reg"}]
        else:
            targets = [c for c in df.columns if df[c].nunique() < 25]
            algos = [{"label": "Random Forest Clf", "value": "rf_clf"}, {"label": "Logistic Reg", "value": "log_reg"}]
        return targets, df.columns.tolist(), algos

    @app.callback(
        Output("exp-id", "data"),
        Output("poll-interval", "disabled"),
        Input("run-btn", "n_clicks"),
        [
            State("model-eval-data-source", "data"), # Correctly listens to the gatekeeper
            State("task-type", "value"), State("target-col", "value"), State("feature-cols", "value"),
            State("algo", "value"), State("trials", "value"), State("split-slider", "value")
        ],
        prevent_initial_call=True
    )
    def launch_exp(_, json_data, task, target, feats, algo, n_trials, split):
        ensure_setup()
        if not all([json_data, target, feats, algo]):
            return dash.no_update, True
        exp_id = str(uuid.uuid4())
        threading.Thread(target=run_automl_job, args=(exp_id, task, json_data, target, feats, algo, n_trials, split, TMP_DIR), daemon=True).start()
        return exp_id, False

    @app.callback(
        Output("results-content", "children"),
        Output("sim-panel", "children"),
        Output("poll-interval", "disabled", allow_duplicate=True),
        Output("eval-alert", "children"),
        Output("eval-alert", "is_open"),
        Input("poll-interval", "n_intervals"),
        [State("exp-id", "data"), State("model-eval-data-source", "data"), State("task-type", "value"), State("target-col", "value")],
        prevent_initial_call=True
    )
    def poll_for_results(_, exp_id, json_data, task, target):
        ensure_setup()
        if not all([exp_id, json_data, task, target]): raise dash.exceptions.PreventUpdate

        error_path = TMP_DIR / f"{exp_id}_error.txt"
        progress_path = TMP_DIR / f"{exp_id}_progress.txt"
        features_path, pipeline_path = TMP_DIR / f"{exp_id}_features.json", TMP_DIR / f"{exp_id}_pipeline.pkl"

        if error_path.exists():
            error_msg = error_path.read_text()
            sim_panel_fail = [html.Small("Experiment failed.", className="text-muted"), html.Div(id="sim-pred", className="mt-3")]
            error_content = html.Div(empty_fig("Experiment Failed"))
            return error_content, sim_panel_fail, True, error_msg, True

        if features_path.exists() and pipeline_path.exists():
            try:
                run = mlflow.search_runs(experiment_names=["DashAutoML"], filter_string=f"tags.`mlflow.runName` = '{exp_id}'").iloc[0]
            except IndexError:
                return dash.no_update, dash.no_update, False, "Experiment queued... waiting for MLflow.", True

            model = mlflow.pyfunc.load_model(f"runs:/{run.run_id}/model")
            df = pd.read_json(io.StringIO(json_data), orient="split")
            feats = json.loads(features_path.read_text())
            X, y = df[feats], df[target]
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
            y_pred = model.predict(X_test)

            if task == "regression":
                fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, title="Model Performance", trendline="ols")
                metrics = {
                    "R-squared (R²)": f"{r2_score(y_test, y_pred):.3f}",
                    "Mean Absolute Error (MAE)": f"{mean_absolute_error(y_test, y_pred):.3f}",
                    "Mean Squared Error (MSE)": f"{mean_squared_error(y_test, y_pred):.3f}",
                }
            else:
                fig = px.imshow(confusion_matrix(y_test, y_pred), text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
                metrics = {
                    "Precision": f"{precision_score(y_test, y_pred, average='weighted'):.3f}",
                    "Recall": f"{recall_score(y_test, y_pred, average='weighted'):.3f}",
                    "F1-Score": f"{f1_score(y_test, y_pred, average='weighted'):.3f}",
                }

            metrics_card_content = []
            for name, value in metrics.items():
                metrics_card_content.extend([html.H5(name, className="card-title"), html.P(value, className="card-text")])
            metrics_card = dbc.Card([dbc.CardHeader("Performance Metrics"), dbc.CardBody(metrics_card_content)])

            results_content = dbc.Row([
                dbc.Col(dcc.Graph(figure=fig), width=8),
                dbc.Col(metrics_card, width=4)
            ])

            sim_controls = []
            for f in feats:
                label = dbc.Label(f)
                control = dcc.Slider(id={"type": "sim-input", "index": f}, min=X[f].min(), max=X[f].max(), value=X[f].mean(), marks=None, tooltip={"placement": "bottom", "always_visible": True}) if pd.api.types.is_numeric_dtype(X[f]) else dcc.Dropdown(id={"type": "sim-input", "index": f}, options=X[f].unique().tolist(), value=X[f].mode()[0])
                sim_controls.append(html.Div([label, control], className="mb-2"))
            sim_panel = sim_controls + [html.Div(id="sim-pred", className="mt-3")]

            return results_content, sim_panel, True, "Experiment finished successfully!", True

        if progress_path.exists():
            progress_text = progress_path.read_text()
            current, total = map(int, progress_text.split(','))
            percent = (current / total) * 100
            progress_bar = html.Div([
                html.H4("Training in Progress...", className="text-center mt-5"),
                html.P(f"Running Trial {current} of {total}", className="text-center"),
                dbc.Progress(value=percent, label=f"{percent:.0f}%", style={"height": "30px"})
            ], className="p-5")
            return progress_bar, dash.no_update, False, "Experiment in progress...", True

        return html.Div(empty_fig("Nothing yet.")), dash.no_update, False, "Experiment queued...", True

    @app.callback(
        Output("sim-pred", "children"),
        Input({"type": "sim-input", "index": ALL}, "value"),
        [State({"type": "sim-input", "index": ALL}, "id"), State("exp-id", "data")],
        prevent_initial_call=True
    )
    def sim(values, ids, exp_id):
        ensure_setup()
        if not all([values, ids, exp_id]): raise dash.no_update
        pipeline_path = TMP_DIR / f"{exp_id}_pipeline.pkl"
        if not pipeline_path.exists(): raise dash.no_update
        pipe = joblib.load(pipeline_path)
        feature_dict = {item['index']: val for item, val in zip(ids, values)}
        sample = pd.DataFrame([feature_dict])
        pred = pipe.predict(sample)[0]
        pred_text = f"{pred:.2f}" if isinstance(pred, (int, float)) else str(pred)
        return dbc.Alert(f"Predicted: {pred_text}", color="info")