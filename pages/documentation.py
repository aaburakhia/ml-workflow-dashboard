import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/documentation')

# Helper function to create a step card
def create_step_card(step_number, title, content):
    return dbc.Card(
        [
            dbc.CardHeader(html.H4(f"Step {step_number}: {title}")),
            dbc.CardBody(content)
        ],
        className="mb-4"
    )

layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            html.H1("How to Use the ML Workflow Dashboard", className="text-center my-4 fw-bold"),
            width=12
        )
    ),
    dbc.Row(
        dbc.Col(
            html.P(
                "This guide will walk you through the complete workflow of the application, from uploading your data to analyzing a trained machine learning model.",
                className="lead text-center mb-4"
            ),
            width=12
        )
    ),

    # Step 1: Home Page
    create_step_card(1, "Upload Your Data on the Home Page", html.Div([
        html.P("The entire process begins on the Home page."),
        html.Ol([
            html.Li("Click the 'Drag and Drop or Select a CSV File' area to open a file browser, or simply drag a CSV file onto it."),
            html.Li("Once uploaded, you will see a success message. The application now holds your raw data in memory, ready for the next steps."),
            html.Li("Use the main navigation bar at the top to proceed to the 'Analysis Modules'.")
        ])
    ])),

    # Step 2: EDA Page
    create_step_card(2, "Explore Your Data with EDA", html.Div([
        html.P("The Exploratory Data Analysis (EDA) page helps you visualize and understand your raw data."),
        html.Ol([
            html.Li("Use the 'Choose Chart Type' dropdown to select a plot, such as a Time Series or a Histogram."),
            html.Li("Based on your choice, new dropdowns will appear. Select the columns from your data that you want to plot."),
            html.Li("The chart will update automatically, giving you a visual sense of your data's patterns and distributions.")
        ])
    ])),

    # Step 3: Preprocessing Page
    create_step_card(3, "Clean Your Data with Preprocessing", html.Div([
        html.P("This page allows you to prepare your data for machine learning. It provides a live 'before and after' preview."),
        html.Ol([
            html.Li("On the left, you'll see a preview of your original, raw data."),
            html.Li("In the 'Preprocessing Controls' panel, you can choose how to handle missing values and whether to scale your numerical data."),
            html.Li("As you change the radio buttons, the 'Processed Data Preview' on the right will instantly update to show you the result of your choices."),
            html.Li("Once you are satisfied with the settings, click the 'Apply & Save Preprocessing' button. This saves the cleaned data and makes it available for the final step.")
        ])
    ])),

    # Step 4: Model Evaluation Page
    create_step_card(4, "Train and Analyze Your Model", html.Div([
        html.P("This is the main MLOps page where you train, evaluate, and analyze a model."),
        html.H5("How to Use This Page:", className="mt-3"),
        html.Ol([
            html.Li("First, configure your experiment in the 'Data & Task' and 'Model' cards on the left. Select your target variable, the features for the model, and the algorithm to use."),
            html.Li("Click 'Launch Experiment'. A real-time progress bar will appear, showing you the status of the AutoML training process."),
            html.Li("When the experiment is finished, the results will appear automatically."),
        ]),
        html.H5("Understanding the Results:", className="mt-3"),
        # --- THIS IS THE CORRECTED BLOCK ---
        html.Ul([
            html.Li([
                html.Strong("Performance Tab:"),
                html.Span(" This shows the main results. You'll see a graph (like a scatter plot of predictions vs. actuals) and a card with key performance metrics like R-squared or Mean Squared Error.")
            ]),
            html.Li([
                html.Strong("Fairness & Drift Tab:"),
                html.Span(" Here you can check your model for bias. Select a 'Sensitive Attribute' (like a category or a binned number) to see if the model's error is higher for one group than another. You can also check for data drift by selecting a time column and a feature to see if its distribution has changed over time.")
            ]),
            html.Li([
                html.Strong("What-if Simulator:"),
                html.Span(" This panel on the left becomes active after a model is trained. You can use the sliders and dropdowns to change the input values and see how the model's prediction changes in real-time.")
            ])
        ])
    ])),

], fluid=True)