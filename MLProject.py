<<<<<<< HEAD
# MLProject.py
import dash
import pandas as pd
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import os

# Load Excel file
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, 'MLresults.xlsx')
print("Attempting to read from:", file_path)
df = pd.read_excel(file_path)

# Create layout
layout = html.Div([
    html.H1('Machine Learning Project', style={'textAlign': 'center', 'color': '#FFF'}),
    html.H2('Class Imbalance in Datasets', style={'textAlign': 'center', 'color': '#FFF'}),
    html.P('The goal of this project is to handle class imbalance (CI) in datasets when performing predictive modelling. The datasets used in this project are imbalanced and the goal is to compare the accuracy of the models before and after applying the CI solution. Handling CI can be acomplished by a number of techniques some of which are as follows,', style={'whiteSpace': 'pre-wrap', 'padding': '20px'}),
    html.Ul([
        html.Li('Synthetic Minority Over-sampling Technique (SMOTE)'),
        html.Li('Random Over Sampling'),
        html.Li('Class Weighting'),
        html.Li('One-Class learning'),
        html.Li('Ensemble techniques')
    ]),
    html.P('The models used are as follows:', style={'whiteSpace': 'pre-wrap', 'padding': '20px'}),
    html.Ul([
        html.Li('K-Nearest Neighbours'),
        html.Li('Logistic Regression'),
        html.Li('Naive Bayes'),
        html.Li('Random Forest'),
        html.Li('XGBoost')
    ]),
    html.P('The datasets used are as follows:', style={'whiteSpace': 'pre-wrap', 'padding': '20px'}),
    html.Ul([
    html.Li([
        'Dataset 1: Celestial Object Classification, 43 features, 3 classes',
        html.Ul([
            html.Li('Class 1: Stars'),
            html.Li('Class 2: Galaxies'),
            html.Li('Class 3: Qausi Stellar Objects (QSO)')
        ])
    ]),
    html.Li([
        'Dataset 2: Fake News Classifcation, 8 features, 3 classes',
        html.Ul([
            html.Li('Class 1: unrelated'),
            html.Li('Class 2: agreed'),
            html.Li('Class 3: disagreed')
        ])
    ]),
    html.Li([
        'Dataset 3: Credit Card Record , 18 features, 8 classes',
        html.Ul([
            html.Li('Class 1: X (no data)'),
            html.Li('Class 2: 0 (no payment delay)'),
            html.Li('Class 3: C (account closed)'),
            html.Li('Class 4: 1 (1 month payment delay)'),
            html.Li('Class 5: 2 (2 months payment delay)'),
            html.Li('Class 6: 3 (3 months payment delay)'),
            html.Li('Class 7: 4 (4 months payment delay)'),
            html.Li('Class 8: 5 (5 months payment delay)'),
        ])
    ])
]),       
    html.P('In the table below you will find three datasets and their respective prediction accuracies based on the sampling technique used.', style={'whiteSpace': 'pre-wrap', 'padding': '20px'}),
    html.H2('Results', style={'textAlign': 'center', 'color': '#FFF'}),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_data={
            'backgroundColor': '#1E1E1E',
            'color': '#FFF',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '16px',
        },
        style_header={
            'backgroundColor': '#333',
            'fontWeight': 'bold',
            'color': '#FFF',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '18px',
        },
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
        },
    ),
    html.H2('Conclusion', style={'textAlign': 'center', 'color': '#FFF'}),
    html.P('After working with these datasets we were able to conclude that for the simpler algorithms, the CI solution takes quite a hit and is adequately affected based on the model, but for the ensemble techniques there is marginal change in accuracy after the CI solution and that too is a worse accuracy than on the imbalanced dataset. This is understandable as ensemble techniques are capable of handling class imbalance with the help of multiple learners.', style={'whiteSpace': 'pre-wrap', 'padding': '20px'})
=======
# MLProject.py
import dash
import pandas as pd
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import os

# Load Excel file
base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, 'MLresults.xlsx')
print("Attempting to read from:", file_path)
df = pd.read_excel(file_path)

# Create layout
layout = html.Div([
    html.H1('Machine Learning Project', style={'textAlign': 'center', 'color': '#FFF'}),
    html.H2('Class Imbalance in Datasets', style={'textAlign': 'center', 'color': '#FFF'}),
    html.P('The goal of this project is to handle class imbalance (CI) in datasets when performing predictive modelling. The datasets used in this project are imbalanced and the goal is to compare the accuracy of the models before and after applying the CI solution. Handling CI can be acomplished by a number of techniques some of which are as follows,', style={'whiteSpace': 'pre-wrap', 'padding': '20px'}),
    html.Ul([
        html.Li('Synthetic Minority Over-sampling Technique (SMOTE)'),
        html.Li('Random Over Sampling'),
        html.Li('Class Weighting'),
        html.Li('One-Class learning'),
        html.Li('Ensemble techniques')
    ]),
    html.P('The models used are as follows:', style={'whiteSpace': 'pre-wrap', 'padding': '20px'}),
    html.Ul([
        html.Li('K-Nearest Neighbours'),
        html.Li('Logistic Regression'),
        html.Li('Naive Bayes'),
        html.Li('Random Forest'),
        html.Li('XGBoost')
    ]),
    html.P('The datasets used are as follows:', style={'whiteSpace': 'pre-wrap', 'padding': '20px'}),
    html.Ul([
    html.Li([
        'Dataset 1: Celestial Object Classification, 43 features, 3 classes',
        html.Ul([
            html.Li('Class 1: Stars'),
            html.Li('Class 2: Galaxies'),
            html.Li('Class 3: Qausi Stellar Objects (QSO)')
        ])
    ]),
    html.Li([
        'Dataset 2: Fake News Classifcation, 8 features, 3 classes',
        html.Ul([
            html.Li('Class 1: unrelated'),
            html.Li('Class 2: agreed'),
            html.Li('Class 3: disagreed')
        ])
    ]),
    html.Li([
        'Dataset 3: Credit Card Record , 18 features, 8 classes',
        html.Ul([
            html.Li('Class 1: X (no data)'),
            html.Li('Class 2: 0 (no payment delay)'),
            html.Li('Class 3: C (account closed)'),
            html.Li('Class 4: 1 (1 month payment delay)'),
            html.Li('Class 5: 2 (2 months payment delay)'),
            html.Li('Class 6: 3 (3 months payment delay)'),
            html.Li('Class 7: 4 (4 months payment delay)'),
            html.Li('Class 8: 5 (5 months payment delay)'),
        ])
    ])
]),       
    html.P('In the table below you will find three datasets and their respective prediction accuracies based on the sampling technique used.', style={'whiteSpace': 'pre-wrap', 'padding': '20px'}),
    html.H2('Results', style={'textAlign': 'center', 'color': '#FFF'}),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_data={
            'backgroundColor': '#1E1E1E',
            'color': '#FFF',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '16px',
        },
        style_header={
            'backgroundColor': '#333',
            'fontWeight': 'bold',
            'color': '#FFF',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '18px',
        },
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
        },
    ),
    html.H2('Conclusion', style={'textAlign': 'center', 'color': '#FFF'}),
    html.P('After working with these datasets we were able to conclude that for the simpler algorithms, the CI solution takes quite a hit and is adequately affected based on the model, but for the ensemble techniques there is marginal change in accuracy after the CI solution and that too is a worse accuracy than on the imbalanced dataset. This is understandable as ensemble techniques are capable of handling class imbalance with the help of multiple learners.', style={'whiteSpace': 'pre-wrap', 'padding': '20px'})
>>>>>>> master
], style={'backgroundColor': '#1E1E1E', 'color': '#FFF', 'fontFamily': 'Arial, sans-serif'})