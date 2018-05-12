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

os.chdir(path + 'images/')
image_filename = 'wordcloud.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

os.chdir(path + 'dash/')
df = UD.Ratios
df2 = UD.df
all_models = UD.all_models
model_options = ['label_lr', 'label_mnb', 'label_svm', 'label_rf', 'label_knn']

#img_file = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/images/wordcloud.png'
#encoded_image = base64.b64encode(open(img_file, 'rb').read())
image_filename = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/images/wordcloud.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())


#image_directory = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/images/wordcloud.png'
#list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
#static_image_route = '/static/'

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

# colors2 = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']
colors2 = ['#B8F7D4', '#835AF1', '#7FA6EE', '#FEBFB3']

# #835AF1 dark blue
# #7FA6EE light blue
# #B8F7D4 green


import numpy as np
df2["com_remv"] = df2["com_remv"].apply(', '.join)
df2["com_remv"] = df2["com_remv"].str.replace(",","").astype(str)

positive = df2[df2["label"]==1]
positive = positive["com_remv"]
negative = df2[df2["label"]==-1]
negative = negative["com_remv"]
neutral = df2[df2["label"]==0]
neutral = neutral["com_remv"]

most_freq_pos = pd.Series(' '.join(positive).lower().split()).value_counts()[:10]
most_freq_neg = pd.Series(' '.join(negative).lower().split()).value_counts()[:10]
most_freq_neu = pd.Series(' '.join(neutral).lower().split()).value_counts()[:10]

# word frequency bar plot

Positive = go.Bar(
            x = most_freq_pos.index,
            y = most_freq_pos.values,
            name="Positive",
            marker=dict(color='#B8F7D4')
        )
Neutral = go.Bar(
            x = most_freq_neu.index,
            y = most_freq_neu.values,
            name="Neutral",
            marker=dict(color='#7FA6EE')
        )
Negative = go.Bar(
            x = most_freq_neg.index,
            y = most_freq_neg.values,
            name="Negative",
            marker=dict(color='#835AF1')
        )

updatemenus = list([
            
            dict(type="buttons",
                 active=-1,
                 buttons=list([   
                    dict(label = 'Positive',
                         method = 'update',
                         args = [{'visible': [True, False, False]},
                                 {'title': 'Positive Comments'}]
                        ),
                    dict(label = 'Neutral',
                         method = 'update',
                         args = [{'visible': [False, True, False]},
                                 {'title': 'Neutral Comments'}]
                        ),
                    dict(label = 'Negative',
                         method = 'update',
                         args = [{'visible': [False, False, True]},
                                 {'title': 'Negative Comments'}]
                        ),
                    dict(label = 'All',
                         method = 'update',
                         args = [{'visible': [True, True, True, True]},
                                 {'title': 'All Comments'}]
                        )
                 ]),
                    pad= {'r': 15, 't': 10}, 
                )
        ])

app = dash.Dash()

'''
-----------------------------------------------------------------
'''

'''
# Video Input Line
    dcc.Input(id='video-input', value='Enter Youtube video URL here', type='text', 
        style={
                'position': 'relative', 
                'width': '600px', 
                'float': 'center', 
                'display': 'inline-block'},
                ),

    html.Div(id='video-input-div', 
        style={
                'position': 'relative', 
                'width': '600px', 
                'float': 'center', 
                'display': 'inline-block'},
                ),
'''

app.layout = html.Div([

# Header
    html.H1(children='A YouTube Web App', 
        style={
            'padding': '10px',
            'text-align': 'center',
            'font-size': '40px'}
        ),

# Pie Chart
    html.Div(
        [
            dcc.Dropdown(
                id="Manager",
                options=[{
                    'label': i,
                    'value': i
                } for i in model_options],
                value='All Models'),
            dcc.Graph(id='funnel-graph')
        ],
            style={
                'float': 'left',
                'width': '40.00%',
                'padding': '10px 10px 10px 0px',
                'height': '300px'}
        ),

# Bar Chart; Right
    html.Div([
        dcc.Graph(
                id='example-graph',
                figure={
                    'data': [Positive, Neutral, Negative],
                    'layout': go.Layout(title='Most Common Words', barmode='stack', showlegend=True,
                            updatemenus=updatemenus)
                        },
                style={
                'float': 'right',
                'width': '55.00%',
                'padding': '42px 0px 10px 10px',
                'height': '500px'
                }
                )
            ])

    #html.H2("WordCloud"),
    #html.Img(src='data:image/png;base64,{}'.format(encoded_image)),
    
    #html.Img(src='/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/images/wordcloud.png', 
    #    style={'width': '500px'})
    #html.Img(src='data:image/png;base64,{}'.format(encoded_image))
    
])

'''
-----------------------------------------------------------------
'''


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
        marker=dict(colors= colors2))

    return {
        'data': [trace],
        'layout':
        go.Layout(
            title='Sentiment Ratios as Predicted by {}'.format(Manager)
            )
    }

my_css_url = "https://github.com/adonovan7/YoutubeAnalysis/blob/master/dash/dash.css"
app.css.append_css({
    "external_url": my_css_url
})


if __name__ == '__main__':
    app.run_server(debug=True)