import dash; import os; import sys; import csv
from dash.dependencies import Input, Output, Event
import dash_core_components as dcc; import dash_html_components as html
import plotly; import flask; import glob; import plotly.plotly as py
import plotly.graph_objs as go; import pandas as pd; import base64
import Classifier as UD
import flask
import glob
import os


path = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/'
os.chdir(path + 'dash/')
data = UD.data # user loaded dataset
df = UD.df # labeled dataset
all_models = UD.all_models # table of average model results for % pos, neg, neu
Ratios = UD.Ratios # % pos, neg, neu for each model
Table = UD.Table # classification for each comment by model

model_options = ['label_mnb', 'label_lr', 'label_xgb']

#mydict = {'label_lr': 'Logistic Regression', 'label_mnb':'Multinomial Naive Bayes',
#'label_svm':'Support Vector Machine', 'label_rf': 'Random Forest', 'label_knn': 'K-Nearest Neighbor'}

mydict = {'label_mnb':'Multinomial Naive Bayes','label_lr': 'Logistic Regression', 'label_xgb': 'Extreme Grandient Boost'}

#colorPalatte= {'positive':'#ef5851', 'neutral': '#cae29a', 'negative':'#437f7c', 'bk':'#c3d8d7', 'white':'#ffffff'}
colorPalatte= {'positive':'#c3d8d7', 'neutral': '#437f7c', 'negative':'#304948', 'bk':'#c3d8d7', 'white':'#ffffff'}
colorP = ['#c3d8d7', '#437f7c', '#304948']
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
            marker=dict(color=colorPalatte["positive"])
        )
Neutral = go.Bar(
            x = most_freq_neu.index,
            y = most_freq_neu.values,
            name="Neutral",
            marker=dict(color=colorPalatte["neutral"])
        )
Negative = go.Bar(
            x = most_freq_neg.index,
            y = most_freq_neg.values,
            name="Negative",
            marker=dict(color=colorPalatte["negative"])
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


image_directory = path + 'dash/images/'
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
static_image_route = '/static/'


app = dash.Dash()

app.layout = html.Div([

# Header
    html.H1(children='YouTube Comment Analyzer',
        style={
            'padding': '8px',
            'text-align': 'center',
            'font-size': '50px'}
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
                'padding': '10px 10px 10px 10px',
                'height': '300px'}
        ),

# Bar Chart; Right
    html.Div([
        dcc.Graph(
                id='bar-graph',
                figure={
                    'data': [Positive, Neutral, Negative],
                    'layout': go.Layout(title='Most Common Words by Sentiment',
                    barmode='stack', showlegend=True,
                            updatemenus=updatemenus)
                        },
                style={
                'float': 'right',
                'width': '55.00%',
                'padding': '5px 10px 10px 10px',
                'height': '500px'
                }
                )
            ]),

# Dropdown Table of Comments
    html.Div([
        dcc.Dropdown(
            id='my-table-dropdown',
            options=[{'label': i, 'value': i}
            for i in ['All Comments', 'Positive', 'Neutral', 'Negative']
            ],value=None),
        html.Div(id='table-container')
        ],
            style={'width': '45%',
            'display': 'inline-block',
            'padding': '0px 5px 5px 10px'}
            ),
# wordclouds
    html.Div([
        dcc.Dropdown(
            id='image-dropdown',
            options=[{'label': i, 'value': i} for i in list_of_images],
            value=list_of_images[0]
        ),
        html.Img(id='image')
    ],
    style={
    'float': 'right',
    'width': '45%',
    'padding': '0px 15px 5px 0px'})


])

# pie chart
@app.callback(
    dash.dependencies.Output('pie-graph', 'figure'),
    [dash.dependencies.Input('MyModel', 'value')])
def update_graph(MyModel):
    if MyModel == "All Models":
        values = [.2,.3,.5]
    else:
        values = list(Ratios[str(MyModel)])

    trace = go.Pie(labels=["Positive","Neutral", "Negative"], values=values, hole=.2,
        name='MyModel', hoverinfo='label+percent',
        textinfo='label + value',textfont=dict(size=10),
        marker=dict(colors= colorP))

    return {
        'data': [trace],
        'layout':
        go.Layout(
            title='Sentiment Ratios as Predicted by {}'.format(MyModel)
            )
    }

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

# wordcloud
@app.callback(
    dash.dependencies.Output('image', 'src'),
    [dash.dependencies.Input('image-dropdown', 'value')])
def update_image_src(value):
    return static_image_route + value

@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)


if __name__ == '__main__':
    app.run_server(debug=True)
