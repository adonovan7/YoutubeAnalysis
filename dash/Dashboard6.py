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
import csv
import pandas as pd
import base64
import Classifier as UD
import os

path = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/'

os.chdir(path + 'images/')
image_filename = 'wordcloud.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

os.chdir(path + 'dash/')
data = UD.data # user loaded dataset
df = UD.df # labeled dataset
all_models = UD.all_models # table of average model results for % pos, neg, neu
Ratios = UD.Ratios # % pos, neg, neu for each model
Table = UD.Table # classification for each comment by model

model_options = ['label_lr', 'label_mnb', 'label_svm', 'label_rf', 'label_knn']

mydict = {'label_lr': 'Logistic Regression', 'label_mnb':'Multinomial Naive Bayes',
'label_svm':'Support Vector Machine', 'label_rf': 'Random Forest', 'label_knn': 'K-Nearest Neighbor'}

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


# extracting comments for each label
positive = UD.positive
negative = UD.negative
neutral = UD.neutral


# most frequent words in each label
most_freq_pos = UD.most_freq_pos
most_freq_neg = UD.most_freq_neg
most_freq_neu = UD.most_freq_neu

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

def generate_table(dataframe, max_rows=10):
    return html.Table(
        [html.Tr([html.Th(col) for col in dataframe.columns])] +
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

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
                id="MyModel",
                options=[{
                    'label': mydict.get(str(i)),
                    'value': i
                } for i in model_options],
                value='All Models'),
            dcc.Graph(id='pie-graph')
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
                id='bar-graph',
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
            ]),
    #html.H2("WordCloud"),
    #html.Img(src='data:image/png;base64,{}'.format(encoded_image)),

    #html.Img(src='/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/images/wordcloud.png',
    #    style={'width': '500px'})
    #html.Img(src='data:image/png;base64,{}'.format(encoded_image))
    html.Div([
        dcc.Dropdown(
            id='my-table-dropdown',
            options=[{'label': i, 'value': i}
            for i in ['All Comments', 'Positive', 'Negative', 'Neutral']
            ],value=None),
        html.Div(id='table-container')
        ],
            style={'width': '49%',
            'display': 'inline-block',
            'padding': '0 20'}
            ),
])

'''
-----------------------------------------------------------------
'''
'''
    html.Div([
        dcc.Graph(
            id='bubble',

            figure={
            'data': go.Scatter(
                x = most_freq_neu.index,
                y = -1.0,
                name="Neutral",
                mode='markers',
                marker=dict(
                    size=most_freq_neu.values,
                    color='#7FA6EE'))
            },

            style={
                'float': 'right',
                'width': '55.00%',
                'padding': '42px 0px 10px 10px',
                'height': '500px'
                }
            )
        ])
'''

# pie chart
@app.callback(
    dash.dependencies.Output('pie-graph', 'figure'),
    [dash.dependencies.Input('MyModel', 'value')])
def update_graph(MyModel):
    if MyModel == "All Models":
        values = [.2,.3,.5]
    else:
        values = list(Ratios[str(MyModel)])

    trace = go.Pie(labels=["Positive", "Negative","Neutral"], values=values, hole=.2,
        name='MyModel', hoverinfo='label+percent',
        textinfo='label + value',textfont=dict(size=20),
        marker=dict(colors= colors2))

    return {
        'data': [trace],
        'layout':
        go.Layout(
            title='Sentiment Ratios as Predicted by {}'.format(MyModel)
            )
    }

'''my_css_url = "https://github.com/adonovan7/YoutubeAnalysis/blob/master/dash/dash.css"
app.css.append_css({
    "external_url": my_css_url
})
'''

# table of comments
@app.callback(
    dash.dependencies.Output('table-container', 'children'),
    [dash.dependencies.Input('my-table-dropdown', 'value')])
def table_update(value):
    simple_df = data[["label","comment"]]
    selected = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
    if value != "All Comments":
        filtered_df = simple_df[simple_df["label"]==selected.get(value)]
    else:
         filtered_df = simple_df
    return generate_table(filtered_df)

if __name__ == '__main__':
    app.run_server(debug=True)
