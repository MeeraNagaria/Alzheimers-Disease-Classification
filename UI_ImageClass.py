import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_dangerously_set_inner_html
import dash_table
import flask
from scipy import stats
import pandas as pd
import numpy as np
import ast
import Image_Classification as IC

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Determining the severity of Alzeihmers through MRI reports'),
    #html.P('Please enter the URL of the MRI image (jpg format):'),
    dcc.Input(id='inputstate', type='text'),    
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    html.Div(id='output'),
        ])

@app.callback(Output(component_id='output', component_property='children'),
              [Input(component_id='submit-button', component_property='n_clicks')],
              [State(component_id='inputstate', component_property='value')])
                #State(component_id='similarity_type', component_property='value')])

 
def getImage(n_clicks, inputstate):

    image_sev = IC.predImage(inputstate)

    return html.Table(
        # Header
        # [html.Tr([html.Th(col) for col in similarity_table.columns])] +

        # Body
        [html.P([image_sev])])
     
    
if __name__ == '__main__':
    app.run_server(port=(8060),debug=False)
    
    
    