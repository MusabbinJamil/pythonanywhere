# share.py
from dash import html, dcc
import pandas as pd
import plotly.express as px
import matplotlib.image as mpimg
import base64
import os

# print("Current working directory:", os.getcwd())
# print("Files in current directory:", os.listdir('.'))

# data = pd.read_csv('Final Project Data.csv')
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, 'Final Project Data.csv')
print("Attempting to read from:", file_path)
data = pd.read_csv(file_path)

# Print column names to see what's available
print("Available columns:", data.columns.tolist())

# Box plot
image_path = os.path.join(base_path, 'image.png')
print("Attempting to read from:", image_path)
fig1 = mpimg.imread(image_path)
fig1_string = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')

# Pie chart - Use the correct column name instead of 'genre'
# For example, if your data has a 'category' column instead:
category_column = 'data_channel' if 'data_channel' in data.columns else data.columns[0]
print(f"Using '{category_column}' for category grouping")
category_shares = data.groupby(category_column)['shares'].sum()
fig2 = px.pie(category_shares, values='shares', names=category_shares.index, template='plotly_dark', title=f'Distribution of Shares by {category_column.capitalize()}')

# Scatter plot
data = data[data['shares'] <= 200000]
fig3 = px.scatter(data, x='content_word_count', y='shares', template='plotly_dark', color_discrete_sequence=['purple'], labels={'x':'Content Word Count', 'y':'Shares'}, title='Relationship between Shares and Content Word Count (Shares <= 200,000)')

layout = html.Div([
    html.H1('Share  Analysis', style={'textAlign': 'center', 'fontSize': '35px', 'marginBottom': '5px', 'color': '#FFF'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.P('This is a box plot of the number of shares the articles received.', style={'textAlign': 'center', 'fontSize': '20px', 'marginTop': '10px', 'color': '#FFF'}),
    html.Img(src='data:image/png;base64,{}'.format(fig1_string), style={'height': '700px', 'width': '700px', 'margin': 'auto'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.P('This is a pie chart of the distribution of shares by category.', style={'textAlign': 'center', 'fontSize': '20px', 'marginTop': '10px', 'color': '#FFF'}),
    dcc.Graph(figure=fig2, style={'height': '700px', 'width': '700px', 'margin': 'auto'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.P('This is a scatter plot of the content word count vs. the number of shares the article received.', style={'textAlign': 'center', 'fontSize': '20px', 'marginTop': '10px', 'color': '#FFF'}),
    dcc.Graph(figure=fig3, style={'height': '700px', 'width': '700px', 'margin': 'auto'}),
], style={'textAlign': 'center'})