import warnings
warnings.simplefilter('ignore')
import dash
from dash import html, dcc, Input, Output
import home
import share
import trend
import sentiment
import about
import comp_int
import prob_rea
import feedback
from MLProject import layout as MLproject
from ApplicationDevelopmentProject import app as ApplicationDevelopmentProject

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.Br(),
        html.H1("UnMashable", style={'textAlign': 'center', 'fontSize': '70px', 'marginBottom': '5px', 'color': '#FFF'}),
        html.P("A Data Analysis Mash-tape!", style={'textAlign': 'center', 'fontSize': '40px', 'marginTop': '10px', 'marginBottom': '5px', 'color': '#FFF'}),
        # html.P("Unraveling news articles, one byte at a time!", style={'textAlign': 'center', 'fontSize': '40px', 'marginTop': '10px', 'color': '#FFF'}),
        html.Hr(style={'width': '100%'}),
    ]),
    html.Br(),
    html.Div([
        dcc.Link('Home', href='/home', style={
            'marginRight': '10px',
            'padding': '10px',
            'backgroundColor': '#3C3C3C',
            'border': '1px solid #FFF',
            'textDecoration': 'none',
            'color': '#F3F3F3'
        }),
        dcc.Link('Application Development Project', href='/ApplicationDevelopmentProject', style={
            'marginRight': '10px',
            'padding': '10px',
            'backgroundColor': '#3C3C3C',
            'border': '1px solid #FFF',
            'textDecoration': 'none',
            'color': '#FFF'
        }),
        dcc.Link('Machine Learning Project', href='/MLProject', style={
            'marginRight': '10px',
            'padding': '10px',
            'backgroundColor': '#3C3C3C',
            'border': '1px solid #FFF',
            'textDecoration': 'none',
            'color': '#FFF'
        }),
        dcc.Link('Computational Intelligence', href='/CompInt', style={
            'marginRight': '10px',
            'padding': '10px',
            'backgroundColor': '#3C3C3C',
            'border': '1px solid #FFF',
            'textDecoration': 'none',
            'color': '#FFF'
        }),
        dcc.Link('Probabilistic Reasoning', href='/ProbReasoning', style={
        'marginRight': '10px',
        'padding': '10px',
        'backgroundColor': '#3C3C3C',
        'border': '1px solid #FFF',
        'textDecoration': 'none',
        'color': '#FFF'
        }),
        dcc.Link('Feedback', href='/feedback', style={  # Add this new link
        'marginRight': '10px',
        'padding': '10px',
        'backgroundColor': '#3C3C3C',
        'border': '1px solid #FFF',
        'textDecoration': 'none',
        'color': '#FFF'
        }),
        dcc.Link('About', href='/about', style={
            'padding': '10px',
            'backgroundColor': '#3C3C3C',
            'border': '1px solid #FFF',
            'textDecoration': 'none',
            'color': '#FFF'
        }),
    ], style={
        'display': 'flex',
        'justifyContent': 'center',
        'backgroundColor': '#1E1E1E',
    }),
    html.Hr(style={'width': '100%'}),
    html.Div(id='page-content'),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
], style={
    'backgroundColor': '#1E1E1E',
    'color': '#FFF'  # White text
})

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/ApplicationDevelopmentProject':
        return html.Div([
            dcc.Link('Shares Analysis', href='/ApplicationDevelopmentProject/share', style={
                'marginRight': '10px',
                'padding': '10px',
                'backgroundColor': '#3C3C3C',
                'border': '1px solid #FFF',
                'textDecoration': 'none',
                'color': '#F3F3F3'
            }),
            dcc.Link('Trend Analysis', href='/ApplicationDevelopmentProject/trend', style={
                'marginRight': '10px',
                'padding': '10px',
                'backgroundColor': '#3C3C3C',
                'border': '1px solid #FFF',
                'textDecoration': 'none',
                'color': '#F3F3F3'
            }),
            dcc.Link('Sentiment Analysis', href='/ApplicationDevelopmentProject/sentiment', style={
                'marginRight': '10px',
                'padding': '10px',
                'backgroundColor': '#3C3C3C',
                'border': '1px solid #FFF',
                'textDecoration': 'none',
                'color': '#F3F3F3'
            })
        ])
    elif pathname == '/ApplicationDevelopmentProject/share':
        return share.layout
    elif pathname == '/ApplicationDevelopmentProject/trend':
        return trend.layout
    elif pathname == '/ApplicationDevelopmentProject/sentiment':
        return sentiment.layout
    elif pathname == '/MLProject':
        return MLproject
    elif pathname == '/CompInt':
        return comp_int.layout
    elif pathname == '/ProbReasoning':
        return prob_rea.layout
    elif pathname == '/feedback':
        return feedback.layout
    elif pathname == '/about':
        return about.layout
    else:
        return home.layout

<<<<<<< HEAD
# if __name__ == '__main__':
#     app.run_server(debug=True)
=======
if __name__ == '__main__':
    app.run_server(debug=True)
>>>>>>> master
