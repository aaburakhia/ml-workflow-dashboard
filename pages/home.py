import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io

dash.register_page(__name__, path='/')

layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H2("Upload Your Data", className="text-center mt-4"),
                width=12)),
    dbc.Row(
        dbc.Col(dcc.Upload(id='upload-data',
                           children=html.Div([
                               'Drag and Drop or ',
                               html.A('Select a CSV File')
                           ]),
                           style={
                               'width': '100%',
                               'height': '60px',
                               'lineHeight': '60px',
                               'borderWidth': '1px',
                               'borderStyle': 'dashed',
                               'borderRadius': '5px',
                               'textAlign': 'center',
                               'margin': '10px'
                           },
                           multiple=False),
                width={
                    "size": 6,
                    "offset": 3
                })),
    dbc.Row(
        dbc.Col(html.Div(id='output-data-upload'),
                width={
                    "size": 6,
                    "offset": 3
                }))
])


@callback(Output('store-data', 'data'),
          Output('output-data-upload', 'children'),
          Output('store-split-indices-json', 'data'),
          Input('upload-data', 'contents'),
          State('upload-data', 'filename'),
          prevent_initial_call=True)
def update_output(contents, filename):
    if contents:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            json_data = df.to_json(date_format='iso', orient='split')
            children = html.Div([
                html.Hr(),
                html.H5(f"Successfully uploaded: {filename}"),
                dbc.Alert("You can now proceed to the analysis modules.",
                          color="success")
            ])
            return json_data, children, None
        except Exception as e:
            error_children = html.Div(
                [dbc.Alert(f"Error processing file: {e}", color="danger")])
            return None, error_children, None
    return no_update, no_update, no_update
