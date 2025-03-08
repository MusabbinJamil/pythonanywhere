# ApplicationDevelopmentProject.py
import dash
from dash import html, dcc, Input, Output
import share, trend, sentiment

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1('Application Development Project'),
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.H1("Application Development Project", style={'textAlign': 'center', 'fontSize': '70px', 'marginBottom': '5px', 'color': '#FFF'}),
        html.Hr(style={'width': '100%'}),
    ]),
    html.Div([
        dcc.Link(
            'Shares Analysis',
            href='/share',
            style={
                'marginRight':'10px',
                'padding': '10px',
                'backgroundColor': '#3C3C3C',
                'border': '1px solid #FFF',
                'textDecoration': 'none',
                'color': '#F3F3F3'
            }
        ),
        dcc.Link(
            'Trend Analysis',
            href='/trend',
            style={
                'backgroundColor': '#3C3C3C',
                'border': '1px solid #FFF',
                'textDecoration': 'none',
                'color': '#FFF',
                'marginRight':'10px',
                'padding': '10px',

            }
        ),
        dcc.Link(
            'Sentiment Analysis',
            href='/sentiment',
            style={
                'marginRight':'10px',
                'padding': '10px',
                'backgroundColor': '#3C3C3C',
                'border': '1px solid #FFF',
                'textDecoration': 'none',
                'color': '#FFF'
            }
        ),
    ], style={
        'display':'flex',
        'justifyContent':'center',
        'backgroundColor': '#1E1E1E',
    }),
    html.Hr(style={'width': '100%'}),
    html.Div(id='page-content'),
], style={
    'backgroundColor': '#1E1E1E',
    'color': '#FFF'  # White text
})

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/share':
        return share.layout
    elif pathname == '/trend':
        return trend.layout
    elif pathname == '/sentiment':
        return sentiment.layout
    else:
        return html.Div([
            html.H2("Welcome to the Application Development Project"),
            html.P("Please select a page from the menu.")
        ])