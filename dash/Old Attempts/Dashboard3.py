# -*- coding: utf-8 -*-

import dash
from dash.dependencies import Input, Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly
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

colors = {
    'background': 'white',
    'graph_background': 'white',
    'text': 'purple',
    'subtext': 'black',
    'blue_pal': 'lightskyblue', 
    'red_pal': 'lightroal', 
    'yellow_pal': 'yellowgreen',
    'grey_pal': 'lightgrey'
}

df = UD.Ratios
model_options = list(df)

app.layout = html.Div([
    html.H2("Seniment Analysis"),
    html.Div(
        [
            dcc.Dropdown(
                id="model_selection",
                options=[{
                    'label': ['LR', 'MNB', 'SVM', 'RF', 'KNN'],
                    'value': i
                } for i in model_options],
                value= 'All Models'
                ),
        ],
        style={'width': '25%',
               'display': 'inline-block'}),

    dcc.Graph(id='pie-graph'),
])

#labels = UD.Ratios.index
#values = UD.Ratios.index

'''
@app.callback(
    dash.dependencies.Output('pie-graph', 'figure'),
    [dash.dependencies.Input('Model', 'value')])
'''
@app.callback(
    dash.dependencies.Output('pie-graph', 'figure'),
    [dash.dependencies.Input('model_selection', 'value')])

def update_figure(Model):
    
    if Model == "All Models":
        df_plot = df.copy()
        values = [1,2,3]
        sent_labs = ["Pos", "Neu", "Neg"]
    else: 
        df_plot = df[str(Model)]
        values = list(df_plot)
        sent_labs = ["Pos", "Neu", "Neg"]

    trace = go.Pie(labels=sent_labs, values=values, name=str(Model))
    '''
    return {
        'data': trace,
        'layout':
        go.Layout(
            title='Sentiment Ratios as Predicted by {}'.format(Model))
    }
    '''

    figure = {
        'data': trace,
        'layout':
            go.Layout(
                title='Sentiment Ratios as Predicted by {}'.format(Model))
    }
    return figure


if __name__ == '__main__':
    app.run_server(debug=True)
