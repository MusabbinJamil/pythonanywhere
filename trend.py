<<<<<<< HEAD
# trend.py
from dash import html, dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Assuming data is your DataFrame
data = pd.read_csv('Final Project Data.csv')
data['date'] = pd.to_datetime(data['date'])
data['month_year'] = data['date'].dt.to_period('M')

# Extract month names from 'month_year' column
data['month_name'] = data['date'].dt.strftime('%B')
data['month_name'] = pd.Categorical(data['month_name'], 
                                    categories=['January', 'February', 'March', 'April', 'May', 'June', 
                                                'July', 'August', 'September', 'October', 'November',
                                                'December'],
                                    ordered=True)

# Group by 'month_year' and calculate mean shares
shares_over_time = data.groupby('month_name')['shares'].mean().reset_index()

# Split data into two subsets for 2013 and 2014
data_2013 = data[data['month_year'].dt.year == 2013]
data_2014 = data[data['month_year'].dt.year == 2014]
# For 2013
shares_over_time_2013 = data_2013.groupby('month_name')['shares'].mean().reset_index()
# For 2014
shares_over_time_2014 = data_2014.groupby('month_name')['shares'].mean().reset_index()


fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=shares_over_time['month_name'], y=shares_over_time['shares'], mode='lines+markers', name='shares'))
fig1.update_layout(title='Trend of Shares Over Time', xaxis_title='Month-Year', yaxis_title='Average Shares', template='plotly_dark')

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=shares_over_time_2013['month_name'], y=shares_over_time_2013['shares'], mode='lines+markers', name='shares',  line=dict(color='#50C878')))
fig2.update_layout(title='Trend of Shares Over Time in 2013', xaxis_title='Month', yaxis_title='Average Shares', template='plotly_dark')

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=shares_over_time_2014['month_name'], y=shares_over_time_2014['shares'], mode='lines+markers', name='shares', line=dict(color='#DA70D6')))
fig3.update_layout(title='Trend of Shares Over Time in 2014', xaxis_title='Month', yaxis_title='Average Shares', template='plotly_dark')

layout = html.Div([
    html.H1('Trend Analysis', style={'textAlign': 'center', 'fontSize': '35px', 'marginBottom': '5px', 'color': '#FFF'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.P('This is a line plot of the average number of shares the articles received over time.', style={'textAlign': 'center', 'fontSize': '20px', 'marginTop': '10px', 'color': '#FFF'}),
    dcc.Graph(figure=fig1, style={'height': '700px', 'width': '700px', 'margin': 'auto'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.P('This is a line plot of the average number of shares the articles received over time in 2013.', style={'textAlign': 'center', 'fontSize': '20px', 'marginTop': '10px', 'color': '#FFF'}),
    dcc.Graph(figure=fig2, style={'height': '700px', 'width': '700px', 'margin': 'auto'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.P('This is a line plot of the average number of shares the articles received over time in 2014.', style={'textAlign': 'center', 'fontSize': '20px', 'marginTop': '10px', 'color': '#FFF'}),
    dcc.Graph(figure=fig3, style={'height': '700px', 'width': '700px', 'margin': 'auto'}),
=======
# trend.py
from dash import html, dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Assuming data is your DataFrame
data = pd.read_csv('Final Project Data.csv')
data['date'] = pd.to_datetime(data['date'])
data['month_year'] = data['date'].dt.to_period('M')

# Extract month names from 'month_year' column
data['month_name'] = data['date'].dt.strftime('%B')
data['month_name'] = pd.Categorical(data['month_name'], 
                                    categories=['January', 'February', 'March', 'April', 'May', 'June', 
                                                'July', 'August', 'September', 'October', 'November',
                                                'December'],
                                    ordered=True)

# Group by 'month_year' and calculate mean shares
shares_over_time = data.groupby('month_name')['shares'].mean().reset_index()

# Split data into two subsets for 2013 and 2014
data_2013 = data[data['month_year'].dt.year == 2013]
data_2014 = data[data['month_year'].dt.year == 2014]
# For 2013
shares_over_time_2013 = data_2013.groupby('month_name')['shares'].mean().reset_index()
# For 2014
shares_over_time_2014 = data_2014.groupby('month_name')['shares'].mean().reset_index()


fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=shares_over_time['month_name'], y=shares_over_time['shares'], mode='lines+markers', name='shares'))
fig1.update_layout(title='Trend of Shares Over Time', xaxis_title='Month-Year', yaxis_title='Average Shares', template='plotly_dark')

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=shares_over_time_2013['month_name'], y=shares_over_time_2013['shares'], mode='lines+markers', name='shares',  line=dict(color='#50C878')))
fig2.update_layout(title='Trend of Shares Over Time in 2013', xaxis_title='Month', yaxis_title='Average Shares', template='plotly_dark')

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=shares_over_time_2014['month_name'], y=shares_over_time_2014['shares'], mode='lines+markers', name='shares', line=dict(color='#DA70D6')))
fig3.update_layout(title='Trend of Shares Over Time in 2014', xaxis_title='Month', yaxis_title='Average Shares', template='plotly_dark')

layout = html.Div([
    html.H1('Trend Analysis', style={'textAlign': 'center', 'fontSize': '35px', 'marginBottom': '5px', 'color': '#FFF'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.P('This is a line plot of the average number of shares the articles received over time.', style={'textAlign': 'center', 'fontSize': '20px', 'marginTop': '10px', 'color': '#FFF'}),
    dcc.Graph(figure=fig1, style={'height': '700px', 'width': '700px', 'margin': 'auto'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.P('This is a line plot of the average number of shares the articles received over time in 2013.', style={'textAlign': 'center', 'fontSize': '20px', 'marginTop': '10px', 'color': '#FFF'}),
    dcc.Graph(figure=fig2, style={'height': '700px', 'width': '700px', 'margin': 'auto'}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.P('This is a line plot of the average number of shares the articles received over time in 2014.', style={'textAlign': 'center', 'fontSize': '20px', 'marginTop': '10px', 'color': '#FFF'}),
    dcc.Graph(figure=fig3, style={'height': '700px', 'width': '700px', 'margin': 'auto'}),
>>>>>>> master
], style={'textAlign': 'center'})