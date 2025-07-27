# ML Workflow Dashboard

An end-to-end, multi-page web application for demonstrating a complete MLOps workflow, from data upload and preprocessing to automated model training, evaluation, and analysis.

![Dashboard Screenshot](https://i.imgur.com/your-screenshot-url.png)
*Note: To add your own screenshot, take a picture of the running app, upload it to a service like [imgur.com](https://imgur.com), and replace the URL above.*

---

## Key Features

*   **End-to-End Workflow:** A guided, step-by-step user experience that mirrors a real-world data science project (Home -> EDA -> Preprocessing -> Model Evaluation).

*   **Interactive EDA:** Visualize and understand the raw dataset with a suite of dynamic plots.

*   **Live Preprocessing:** A powerful interface with a "before and after" preview for cleaning data, including handling missing values and feature scaling.

*   **Automated Model Training:** Uses **Optuna** for hyperparameter tuning to find the best model automatically, with a real-time progress bar providing clear feedback.

*   **Experiment Tracking:** Integrated with **MLflow** to log all experiment parameters, metrics, and model artifacts for complete reproducibility.

*   **Robust MLOps Analysis:**
    *   **Fairness Analysis:** Check for model bias by analyzing performance across different subgroups of your data (including binned numerical features).
    *   **Data Drift Analysis:** Visualize how the distribution of your data changes over time to identify potential issues with model relevance.

*   **What-if Simulator:** An interactive panel to probe the trained model and see how its predictions change based on your inputs in real-time.

## How to Run This Project

This project is designed to run seamlessly in a Replit environment.

1.  **Fork the Repository:** Import this GitHub repository into your own Replit account.
2.  **Install Packages:** The Replit environment will automatically install all required packages based on the `pyproject.toml` file.
3.  **Run the App:** Click the main "Run" button at the top of the workspace. This will execute `python app.py` and start the web server.
4.  **Interact:** The live application will appear in the "Preview" window.

## Project Structure

This application follows a scalable, modular structure based on the principle of "Separation of Concerns."
/
├── app.py # Main Dash app definition, layout, and navbar
├── logic/ # Core, non-web-related Python logic
│ ├── ml_jobs.py # The AutoML training function
│ └── preprocessing_jobs.py # The data cleaning function
└── pages/ # Each .py file here is a page in the app
├── documentation.py
├── eda.py
├── home.py
├── model_evaluation.py # Entry point for the model evaluation page
├── preprocessing.py # Entry point for the preprocessing page
└── me_components/ # Components for the Model Evaluation page
│ ├── layout.py
│ └── callbacks.py
└── pp_components/ # Components for the Preprocessing page
├── layout.py
└── callbacks.py


## Technologies Used

*   **Dash & Plotly:** For the interactive web interface and data visualization.
*   **Dash Bootstrap Components:** For professional styling and layout.
*   **Pandas:** For data manipulation and analysis.
*   **Scikit-learn:** For machine learning pipelines and metrics.
*   **Optuna:** For automated hyperparameter optimization.
*   **MLflow:** For experiment tracking and MLOps.