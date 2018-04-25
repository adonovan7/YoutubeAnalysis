# -*- coding: utf-8 -*-

import dash
from dash.dependencies import Input, Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import pandas as pd
import base64
from textblob import TextBlob
from collections import deque

import Variables as yt

# https://plot.ly/python/subplots/
# https://github.com/plotly
# https://codepen.io/chriddyp/pen/bWLwgP
# https://dash.plot.ly/interactive-graphing for buttons to actually work
# https://www.youtube.com/watch?v=yfWJXkySfe0

df = yt.df.ix[:, ['label', 'comment']]

'''
def update_pie_value(video):
    video.groupby(['label']).size()
    neg_sent = video.loc[df['label'] < 0]
    zer_sent = video.loc[df['label'] == 0]
    pos_sent = video.loc[df['label'] > 0]
    d1 = len(neg_sent); d2 = len(zer_sent); d3 = len(pos_sent)

    return dcc.Graph(
        id='pie1',
        figure={
            'data': [
                {'x': df.index, 'y': df.Close, 'type': 'line', 'name': input_data},
            ],
            'layout': {
                'title': input_data
            }
        }
    )
'''
analysis = TextBlob("TextBlob sure looks like it has some interesting features!")


df.groupby(['label']).size()
neg_sent = df.loc[df['label'] < 0]
zer_sent = df.loc[df['label'] == 0]
pos_sent = df.loc[df['label'] > 0]
d1 = len(neg_sent); d2 = len(zer_sent); d3 = len(pos_sent)

VideoName = "OK GO"

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


X = deque(maxlen=20)
X.append(1)
Y = deque(maxlen=20)
Y.append(1)

app = dash.Dash()

'''
Dash apps composed of two parts:
1. Layout
2. Interactive Components

* can build components with JS and React.js
'''

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
    ## html. and dcc. are components of the layout tree
    dcc.Tabs(
        tabs=[
            {'label': 'Video {}'.format(i), 'value': i} for i in range(1, 4)
        ],
        value=1,
        id='tabs'
    ), 
    html.Div(id='tab-output'),

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

    #dcc.Textarea(
    #    id='my-id',
	#    placeholder='Enter a value...',
	#    value='Enter Youtube video URL here',
	#    style={'height': '8px', 
	#    'width': '800px', 
	#    'textAlign': 'left', 'padding': 8}
	#	),  
    #dcc.Input(id='my-id', value='initial value', type='text'),
    dcc.Input(id='video-input', value='Enter Youtube video URL here', type='text', 
        style={'width': '600px'}),
    html.Div(id='video-input-div'),

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
     #'vertical-align': 'middle'
     #html.Img(src='data:image/png;base64,{}'.format(encoded_image))
    html.H4(children='Comments for %s' % VideoName),
    generate_table(df),


    html.Div([
		    dcc.Graph(
		                id='pie1',       
		                figure = {
		                    'data': [
		                        {
		                            'labels': ['Negative','Neutral', "Positive"],
		                            'values': [d1, d2, d3],
		                            'type': 'pie',
		                            'name': 'Video Chosen',
		                            'marker': {'colors': ['lightskyblue', 'lightcoral', 'yellowgreen']},
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
							            'marker': { 'color': ['yellowgreen', 'lightcoral',
							               'lightskyblue', 'orange', 'purple', 'lightgreen']},
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
     	], style= {'color': colors['text'], 'textAlign': 'center'}), 

     html.H2('Live Youtube Sentiment'),
        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1*1000
        ),

])

@app.callback(Output('tab-output', 'children'), [Input('tabs', 'value')])
def display_content(value):
    data = [
        {
		    'labels': ['Negative','Neutral', "Positive"],
		    'values': [d1, d2, d3],
		    'type': 'pie',
		    'name': 'Video Chosen',
		    'marker': {'colors': ['yellowgreen', 'lightcoral', 'lightskyblue']},
		    'hoverinfo':'label+percent+name',
		                        },
        {{
			'x': ['MNB','SGD','LR', 'LSV', 'Bag', 'RF'],
			'y': [yt.mnb_acc, yt.sgd_acc, yt.sgd_acc, yt.svm_acc,  yt.bag_acc, yt.rf_acc],
			'type': 'bar',
			'marker': { 'color': ['lightskyblue', 'lightcoral',
							        'yellowgreen', 'orange', 'purple', 'lightgreen']},
			'hoverinfo':'x+y',
							        },
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

@app.callback(
    Output(component_id='video-input-div', component_property='children'),
    [Input(component_id='video-input', component_property='value')]
)

# live sentiment app
@app.callback(Output('live-graph', 'figure'),
              events=[Event('graph-update', 'interval')])


# text input

def update_output_div(input_value):
    return 'Conducting sentiment analysis on the "{}" video'.format(input_value)

# sentiment scatter

def update_graph_scatter():
    try:
        conn = sqlite3.connect('twitter.db')
        c = conn.cursor()
        df = pd.read_sql("SELECT * FROM sentiment WHERE tweet LIKE '%olympic%' ORDER BY unix DESC LIMIT 1000", conn)
        df.sort_values('unix', inplace=True)
        df['sentiment_smoothed'] = df['sentiment'].rolling(int(len(df)/5)).mean()
        df.dropna(inplace=True)

        X = df.unix.values[-100:]
        Y = df.sentiment_smoothed.values[-100:]

        data = plotly.graph_objs.Scatter(
                x=X,
                y=Y,
                name='Scatter',
                mode= 'lines+markers'
                )

        return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                    yaxis=dict(range=[min(Y),max(Y)]),)}

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')



if __name__ == '__main__':
    app.run_server(debug=True)
