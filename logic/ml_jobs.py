# logic/ml_jobs.py

import pandas as pd
import numpy as np
import io, json, traceback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.base import clone
import optuna
import mlflow.sklearn
import joblib

def run_automl_job(exp_id, task, json_data, target, feats, algo, n_trials, split, tmp_dir):
    """
    This function runs the entire AutoML job and reports progress to a file.
    """
    progress_file = tmp_dir / f"{exp_id}_progress.txt"
    try:
        df = pd.read_json(io.StringIO(json_data), orient="split").dropna(subset=[target] + feats)
        X, y = df[feats], df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - split) / 100, random_state=42)
        numeric, cat = X.select_dtypes(include=np.number).columns.tolist(), X.select_dtypes(exclude=np.number).columns.tolist()
        transformers = []
        if len(numeric): transformers.append(("num", StandardScaler(), numeric))
        if len(cat): transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
        prep = ColumnTransformer(transformers=transformers)
        MODELS = {
            "rf_reg": RandomForestRegressor(random_state=42), "lin_reg": LinearRegression(),
            "rf_clf": RandomForestClassifier(random_state=42), "log_reg": LogisticRegression(max_iter=1000, random_state=42)
        }
        base_model = MODELS[algo]

        def objective(trial):
            # --- WRITE PROGRESS TO FILE ---
            # trial.number is 0-indexed, so we add 1
            progress_file.write_text(f"{trial.number + 1},{n_trials}")

            params = {}
            if "rf" in algo:
                params = dict(n_estimators=trial.suggest_int("n_estimators", 50, 300), max_depth=trial.suggest_int("max_depth", 3, 10), min_samples_split=trial.suggest_int("min_samples_split", 2, 10))
            model = clone(base_model).set_params(**params)
            pipe = Pipeline([("prep", prep), ("model", model)])
            pipe.fit(X_train, y_train)
            return mean_absolute_error(y_test, pipe.predict(X_test)) if task == "regression" else -accuracy_score(y_test, pipe.predict(X_test))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        best_pipe = Pipeline([("prep", prep), ("model", clone(base_model).set_params(**study.best_params))]).fit(X_train, y_train)

        mlflow.set_experiment("DashAutoML")
        with mlflow.start_run(run_name=exp_id):
            mlflow.log_params(study.best_params)
            mlflow.log_metrics({"metric": study.best_value})
            mlflow.sklearn.log_model(best_pipe, "model")

        joblib.dump(best_pipe, tmp_dir / f"{exp_id}_pipeline.pkl")
        (tmp_dir / f"{exp_id}_features.json").write_text(json.dumps(feats))

    except Exception as e:
        tb = traceback.format_exc()
        (tmp_dir / f"{exp_id}_error.txt").write_text(f"An error occurred:\n{e}\n\nTraceback:\n{tb}")
    finally:
        # --- CLEANUP: Remove the progress file when done ---
        if progress_file.exists():
            progress_file.unlink()