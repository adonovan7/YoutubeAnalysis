import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import pandas as pd
import base64

from textblob import TextBlob
from collections import deque
from dash.dependencies import Input, Output, Event

#import Variables as yt
#import UserDataModel as UD

import dash
import random

path = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/'
sys.path.insert(0, path +'scripts')
import UserDataModel as UD


app = dash.Dash(__name__)
app.layout = html.Div(
    [   html.H2('Comment Sentiments Ratios'),

        dcc.Input(id='URL', value='Paste URL here', type='text'),

        dcc.Graph(id='live-graph', animate=False),

        dcc.Interval(
            id='graph-update',
            interval=1*1000
        ),

    ]
)

@app.callback(Output('live-graph', 'figure'),
              [Input(component_id='URL', component_property='value')],
              events=[Event('graph-update', 'interval')])

path = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/'

def mySubprocess(vidName, vidLink):
		#print("\nURL for %s video: " %vidName, vidLink)
		args = 'Python3 apiCall.py --c --videourl=' + vidLink + ' >> ' + path + 'data/data.csv'
		#print('Runing URL through API Call.')
		#print('Hint: Press ^C to quit after a few minutes (wait longer if you would like more comments). \n')
		subprocess.run(args, shell=True)
		sys.exit(1)

def update_graph_scatter(URL):
    try:
		mySubprocess("personal", URL)
	df = UD.Ratios

        data = plotly.graph_objs.Pie(
                x=X,
                y=Y,
                name='Scatter',
                mode= 'lines+markers'
                )

        return {'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                    yaxis=dict(range=[min(Y),max(Y)]),
                                                    title='Term: {}'.format(sentiment_term))}

    except Exception as e:
        with open('errors.txt','a') as f:
            f.write(str(e))
            f.write('\n')

if __name__ == '__main__':
    app.run_server(debug=True)