# -*- coding: utf-8 -*-

import dash
from dash.dependencies import Input, Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly

import plotly.plotly as py
import plotly.graph_objs as go
import sys
import os
import csv
import pandas as pd

import UD as UD

path = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/'
os.chdir(path + 'dash/')
#sys.path.insert(0, path +'dash')

app = dash.Dash()

df = UD.Ratios
all_models = UD.all_models
model_options = list(df)


app.layout = html.Div([
    html.H2("Seniment Analysis"),
    html.Div(
        [
            dcc.Dropdown(
                id="model_selection",
                options=[{
                    'label': i,
                    'value': str(i)
                } for i in model_options],
                value= 'All Models'
                ),
        ],
        style={'width': '25%',
               'display': 'inline-block'}),

    dcc.Graph(id='pie-graph'),
])

@app.callback(
    dash.dependencies.Output('pie-graph', 'figure'),
    [dash.dependencies.Input('model_selection', 'value')])
def update_graph(Model):
    if Model == "All Models":
        values = [0.2, 0.3, 0.5]
    else:
        df_plot = df[str(Model)]
        values= list(df_plot)

    trace = go.Pie(labels=["Positive", "Negative","Neutral"], values=values, name='MyModel')
    
    return {
        'data': trace,
        'layout':
        go.Layout(
            title='Sentiment Ratios as Predicted by {}'.format(Model))
    }

if __name__ == '__main__':
    app.run_server(debug=True)
