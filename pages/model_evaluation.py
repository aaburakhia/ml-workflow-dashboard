# pages/model_evaluation.py

import dash
from .me_components.layout import evaluation_layout
from .me_components.callbacks import register_callbacks

# 1. Register the page
dash.register_page(__name__, path='/model-evaluation')

# 2. Define the layout for this page
layout = evaluation_layout

# 3. Register the callbacks for this page
register_callbacks(dash.get_app())