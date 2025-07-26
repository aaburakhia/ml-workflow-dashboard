import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

# Initialize the Dash app, using Bootstrap for styling and enabling the pages module
app = Dash(__name__,
           use_pages=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)

# Define the navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("EDA", href="/eda"),
                dbc.DropdownMenuItem("Preprocessing", href="/preprocessing"),
                dbc.DropdownMenuItem("Model Evaluation",
                                     href="/model-evaluation"),
            ],
            nav=True,
            in_navbar=True,
            label="Analysis Modules",
        ),
        dbc.NavItem(dbc.NavLink("Documentation", href="/documentation")),
    ],
    brand="ML Workflow Dashboard",
    brand_href="/",
    color="primary",
    dark=True,
)

# Define the main layout of the app
app.layout = html.Div([
    navbar,
    # This is where the content of each page will be rendered
    dash.page_container,

    # --- Storage Components ---
    # This is the central, single source of truth for the uploaded data.
    dcc.Store(id='store-data', storage_type='session'),

    # These are the new storage units our advanced module will need.
    # They are empty by default and live here in the main layout
    # so they are always available to any page that needs them.
    dcc.Store(id='store-split-indices-json', storage_type='session'),

    # --- ADD THIS NEW LINE ---
    # This will hold the data after it has been cleaned on the Preprocessing page.
    dcc.Store(id='store-processed-data', storage_type='session'),
])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
