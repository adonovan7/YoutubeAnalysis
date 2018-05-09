# -*- coding: utf-8 -*-

import dash
from dash.dependencies import Input, Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import pandas as pd
import base64

import Variables as yt

# https://plot.ly/python/subplots/
# https://github.com/plotly
# https://codepen.io/chriddyp/pen/bWLwgP
# https://dash.plot.ly/interactive-graphing for buttons to actually work
df = yt.df

df.groupby(['label']).size()
neg_sent = df.loc[df['label'] < 0]
zer_sent = df.loc[df['label'] == 0]
pos_sent = df.loc[df['label'] > 0]
d1 = len(neg_sent); d2 = len(zer_sent); d3 = len(pos_sent)


def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

image_filename = 'wordcloud.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())



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
#lightskyblue
#lightcoral
#yellowgreen



app.layout = html.Div(style={'backgroundColor': colors['background'],'columnCount': 1}, children=[
    dcc.Tabs(
        tabs=[
            {'label': 'Video {}'.format(i), 'value': i} for i in range(1, 4)
        ],
        value=1,
        id='tabs'
    ), 

    html.H1(
        children='Youtube Video Analyzer',
        style={
            'textAlign': 'center',
            'font-family': 'Dosis',
            'font-size': '32x',
            'color': colors['text'], 
            'padding': 0
        }
    ),

    html.Div(children='A web application for analyzing comments on Youtube videos', style={
        'textAlign': 'center',
        'color': colors['subtext'], 
        'padding': 8
    }),

   	html.Label('URL: ', style={
            'font-family': 'Dosis',
            'color': colors['text']
        }),

    dcc.Textarea(
	    placeholder='Enter a value...',
	    value='Enter Youtube Video URL here',
	    style={'height': '8px', 
	    'width': '800px', 
	    'textAlign': 'left', 'padding': 8}
		),  


	dcc.Checklist(
	        options=[
	            {'label': 'Positive', 'value': 'NYC'},
	            {'label': u'Negative', 'value': 'MTL'},
	            {'label': 'Neutral', 'value': 'SF'}
	        ],
	        values=['NYC', 'MTL', 'SF'],
	        style={'textAlign': 'center', 'color': colors['subtext'], 
	        'padding': 0}
	    ),

    html.Div([
		    dcc.Graph(
		                id='pie',       
		                figure = {
		                    'data': [
		                        {
		                            'labels': ['Negative','Neutral', "Positive"],
		                            'values': [d1, d2, d3],
		                            'type': 'pie',
		                            'name': 'Video Chosen',
		                            'marker': {'colors': ['yellowgreen', 'lightcoral', 'lightskyblue']},
		                            'hoverinfo':'label+percent+name',
		                        }
		                    
		                    ],
		                    'layout': {
				                'title': 'Comment Sentiment Ratios',
				                'showlegend': True,
				                'plot_bgcolor': colors['graph_background'],
				                'paper_bgcolor': colors['background'],
				                'font': {
				                    'color': colors['text']
				                }
				                }
		                }
		            )
		    ], style= {'width': '49%', 
		    'display': 'inline-block', 
		    'color': 'black'}),
    
    html.Div([
		    dcc.Graph(
		                id='pie2',       
		                figure = {
							    'data': [
							        {
							            'x': ['MNB','SGD','LR', 'LSV', 'Bag', 'RF'],
							            'y': [yt.mnb_acc, yt.sgd_acc, yt.sgd_acc, yt.svm_acc,  yt.bag_acc, yt.rf_acc],
							            'type': 'bar',
							            'marker': { 'color': ['lightskyblue', 'lightcoral',
							               'yellowgreen', 'orange', 'purple', 'lightgreen']},
							            'hoverinfo':'x+y',
							        }
							    ],
							    'layout': {'title': 'Accuracy of Classification Models',
							               'showlegend': False,
							                'plot_bgcolor': colors['graph_background'],
							                'paper_bgcolor': colors['background'],
							                'font': {
							                    'color': colors['text']}}
							}
		            )
		    ], style= {'width': '49%', 
		    'display': 'inline-block', 
		    'color': colors['text']}),  
#'vertical-align': 'middle'
     #html.Img(src='data:image/png;base64,{}'.format(encoded_image))
     html.H1(
        children='WordCloud of Video:',
        style={
            'textAlign': 'center',
            'font-family': 'Dosis',
            'color': colors['text'], 
            'padding': 0
        }
    ),
     html.Div([
     	html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
     	], style= {'color': colors['text'], 'textAlign': 'center'})

])

@app.callback(Output('tab-output', 'children'), [Input('tabs', 'value')])
def display_content(value):
    data = [
        {
            'x': [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
                  2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
            'y': [219, 146, 112, 127, 124, 180, 236, 207, 236, 263,
                  350, 430, 474, 526, 488, 537, 500, 439],
            'name': 'Rest of world',
            'marker': {
                'color': 'rgb(55, 83, 109)'
            },
            'type': ['bar', 'scatter', 'box'][int(value) % 3]
        },
        {
            'x': [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
                  2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
            'y': [16, 13, 10, 11, 28, 37, 43, 55, 56, 88, 105, 156, 270,
                  299, 340, 403, 549, 499],
            'name': 'China',
            'marker': {
                'color': 'rgb(26, 118, 255)'
            },
            'type': ['bar', 'scatter', 'box'][int(value) % 3]
        }
    ]

    return html.Div([
        dcc.Graph(
            id='graph',
            figure={
                'data': data,
                'layout': {
                    'margin': {
                        'l': 30,
                        'r': 0,
                        'b': 30,
                        't': 0
                    },
                    'legend': {'x': 0, 'y': 1}
                }
            }
        ),
        html.Div(' '.join(get_sentences(10)))
    ])
if __name__ == '__main__':
    app.run_server(debug=True)
