from dash import html

layout = html.Div([
    html.H1("About UnMashable", style={'textAlign': 'center'}),
    html.P(
        "UnMashable is a project by a team of data enthusiasts who are passionate about making sense of the world's news.",
        style={'textAlign': 'center'}
    ),
    html.P(
        "Our team is composed of two members who believe in the power of data to inform and inspire.",
        style={'textAlign': 'center'}
    ),
    html.P(
        "We're always looking for new challenges and opportunities, so don't hesitate to get in touch!",
        style={'textAlign': 'center'}
    ),
    html.Br(),
    html.P("Members : Musab and Hussain", style={'textAlign': 'center'}),
    html.P("ERPs    : 29409 and 29410", style={'textAlign': 'center'}),
    html.P([
    "Email1: ",
    html.A("m.jamil.29409@khi.iba.edu.pk", href="mailto:m.jamil.29409@khi.iba.edu.pk", style={'color': 'blue'})
], style={'textAlign': 'center'}),
    html.P([
    "Email2: ",
    html.A("h.diwan.29410@khi.iba.edu.pk", href="mailto:h.diwan.29410@khi.iba.edu.pk", style={'color': 'blue'})
], style={'textAlign': 'center'}),
    html.P(children=[
        "Link to Mashable dataset: ",
        html.A("Online News Popularity",
               href="https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity",
               target="_blank")
    ],
    style={'textAlign': 'center'}
    ),
    html.P(children=[
            "Link to Kaggle dataset (N/A at the moment): ",
            html.A("Fake News Classification",
                href="",
                target="_blank")
        ],
        style={'textAlign': 'center'}
    )
])