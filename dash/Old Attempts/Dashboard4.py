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

df = UD.Ratios
model_options = list(df)

#df["Sentiment"] = ["Positive", "Negative","Neutral"]


# mgr_options = df["Manager"].unique()

app = dash.Dash()

app.layout = html.Div([
    html.H2("Sales Funnel Report"),
    html.Div(
        [
            dcc.Dropdown(
                id="Manager",
                options=[{
                    'label': i,
                    'value': i
                } for i in model_options],
                value='All Managers'),
        ],
        style={'width': '25%',
               'display': 'inline-block'}),
    dcc.Graph(id='funnel-graph'),
])


@app.callback(
    dash.dependencies.Output('funnel-graph', 'figure'),
    [dash.dependencies.Input('Manager', 'value')])

def update_graph(Manager):
    if Manager == "All Managers":
        df_plot = df["label_lr"]
    else:
        df_plot = df[str(Manager)]

    trace = go.Pie(labels=["Positive", "Negative","Neutral"], values=list(df_plot), name='MyModel')

    return {
        'data': trace,
        'layout':
        go.Layout(
            title='Customer Order Status for {}'.format(Manager))
    }


if __name__ == '__main__':
    app.run_server(debug=True)