# sentiment.py
from dash import html, dcc
import pandas as pd
import plotly.express as px

data = pd.read_csv('Final Project Data.csv')

fig1 = px.scatter(data, x='title_sentiment_polarity', y='shares', template='plotly_dark', color_discrete_sequence=['blue'])
fig2 = px.scatter(data, x='avg_positive_polarity', y='shares', template='plotly_dark', color_discrete_sequence=['green'])
fig3 = px.scatter(data, x='avg_negative_polarity', y='shares', template='plotly_dark', color_discrete_sequence=['red'])

layout = html.Div([
    html.H1('Sentiment Analysis', style={'textAlign': 'center', 'fontSize': '35px', 'marginBottom': '5px', 'color': '#FFF'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.P('This is a scatter plot of the title sentiment polarity vs. the number of shares the article received.', style={'textAlign': 'center', 'fontSize': '20px', 'marginTop': '10px', 'color': '#FFF'}),
    dcc.Graph(figure=fig1, style={'height': '700px', 'width': '700px', 'margin': 'auto'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.P('This is a scatter plot of the average positive polarity of the articles vs. the number of shares the article received.', style={'textAlign': 'center', 'fontSize': '20px', 'marginTop': '10px', 'color': '#FFF'}),
    dcc.Graph(figure=fig2, style={'height': '700px', 'width': '700px', 'margin': 'auto'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.P('This is a scatter plot of the average negative polarity of the articles vs. the number of shares the article received.', style={'textAlign': 'center', 'fontSize': '20px', 'marginTop': '10px', 'color': '#FFF'}),
    dcc.Graph(figure=fig3, style={'height': '700px', 'width': '700px', 'margin': 'auto'}),
], style={'textAlign': 'center'})