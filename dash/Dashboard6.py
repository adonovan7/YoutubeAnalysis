import dash
from dash.dependencies import Input, Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly

import flask
import glob

import plotly.plotly as py
import plotly.graph_objs as go
import sys
import os
import csv
import pandas as pd
import base64

import UD as UD


path = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/'
os.chdir(path + 'dash/')
df = UD.Ratios
all_models = UD.all_models
model_options = ['label_lr', 'label_mnb', 'label_svm', 'label_rf', 'label_knn']

#img_file = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/images/wordcloud.png'
#encoded_image = base64.b64encode(open(img_file, 'rb').read())
image_filename = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/images/wordcloud.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())


image_directory = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/images/wordcloud.png'
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
static_image_route = '/static/'

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

colors2 = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']

app = dash.Dash()

app.layout = html.Div([
    html.H1(children='A web application for analyzing comments on Youtube videos', style={
        'textAlign': 'center',
        'color': colors['text'], 
        'padding': 8
    }),

    dcc.Input(id='video-input', value='Enter Youtube video URL here', type='text', 
        style={'width': '600px'}),

    html.Div(id='video-input-div'),

    html.H2("Sentiment Ratios by Model"),

    html.Div(
        [
            dcc.Dropdown(
                id="Manager",
                options=[{
                    'label': i,
                    'value': i
                } for i in model_options],
                value='All Models'),
        ],
        style={'width': '25%',
               'display': 'inline-block'}),

    dcc.Graph(id='funnel-graph', 
        style={'display': 'left'}),

    html.H2("WordCloud"),

    html.Img(src='data:image/png;base64,{}'.format(encoded_image))
    
    #html.Img(src='/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/images/wordcloud.png', 
    #    style={'width': '500px'})
    #html.Img(src='data:image/png;base64,{}'.format(encoded_image))
    ,
])


@app.callback(
    dash.dependencies.Output('funnel-graph', 'figure'),
    [dash.dependencies.Input('Manager', 'value')])
def update_graph(Manager):
    if Manager == "All Models":
        values = [.2,.3,.5]
    else:
        values = list(df[str(Manager)])

    trace = go.Pie(labels=["Positive", "Negative","Neutral"], values=values, 
        name='MyModel', hoverinfo='label+percent', 
        textinfo='label + value',textfont=dict(size=20),
        marker=dict(colors=colors2))

    return {
        'data': [trace],
        'layout':
        go.Layout(
            title='Sentiment Ratios as Predicted by {}'.format(Manager)
            )
    }


if __name__ == '__main__':
    app.run_server(debug=True)