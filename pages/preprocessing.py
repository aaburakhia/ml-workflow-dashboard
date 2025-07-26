# pages/preprocessing.py

import dash
from .pp_components.layout import preprocessing_layout
from .pp_components.callbacks import register_callbacks

# 1. Register the page with a URL
dash.register_page(__name__, path='/preprocessing')

# 2. Define the layout for this page
layout = preprocessing_layout

# 3. Register the callbacks for this page
register_callbacks(dash.get_app())