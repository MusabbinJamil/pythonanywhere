import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
import numpy as np
from evo_algo import evolutionary_algorithm

layout = html.Div([
    html.H1("Computational Intelligence - Evolutionary Algorithm", 
            style={'textAlign': 'center', 'color': '#FFF'}),
    
    html.Div([
        # Input fields
        html.Div([
            html.Label("Fitness Function:", style={'color': '#FFF'}),
            dcc.Input(
                id='fitness-function',
                type='text',
                placeholder='e.g., x**2 + y**2',
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            
            html.Label("Population Size:", style={'color': '#FFF'}),
            dcc.Input(
                id='pop-size',
                type='number',
                value=10,
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            
            html.Label("(x1, y1): ", style={'color': '#FFF'}),
            dcc.Input(
                id='x1-min',
                type='number',
                value=-5,
                placeholder='Min',
                style={'width': '20%', 'marginRight': '10%'}
            ),
            dcc.Input(
                id='x1-max',
                type='number',
                value=5,
                placeholder='Max',
                style={'width': '20%'}
            ),

            html.Br(),
            
            html.Label("(x2, y2): ", style={'color': '#FFF'}),
            dcc.Input(
                id='x2-min',
                type='number',
                value=-5,
                placeholder='Min',
                style={'width': '20%', 'marginRight': '10%'}
            ),

            dcc.Input(
                id='x2-max',
                type='number',
                value=5,
                placeholder='Max',
                style={'width': '20%'}
            ),

            html.Br(),
            
            html.Label("Parent Selection:", style={'color': '#FFF'}),
            dcc.Dropdown(
                id='parent-selection',
                options=[
                    {'label': 'Roulette Wheel', 'value': 'roulette'},
                    {'label': 'Tournament', 'value': 'tournament'},
                    {'label': 'Rank Based', 'value': 'rank'}
                ],
                value='roulette',
                style={'width': '100%', 'marginBottom': '10px'}
            ),

            
            html.Label("Survival Selection:", style={'color': '#FFF'}),
            dcc.Dropdown(
                id='survival-selection',
                options=[
                    {'label': 'Truncation', 'value': 'truncation'},
                    {'label': 'Tournament', 'value': 'tournament'}
                ],
                value='truncation',
                style={'width': '100%', 'marginBottom': '10px'}
            ),

            html.Label("Crossover Type:", style={'color': '#FFF'}),
            dcc.Dropdown(
                id='crossover-type',
                options=[
                    {'label': 'Arithmetic Crossover', 'value': 'arithmetic'},
                    {'label': 'Swap Crossover', 'value': 'swap'},
                    {'label': 'Blend Crossover', 'value': 'blend'}
                ],
                value='arithmetic',
                style={'width': '100%', 'marginBottom': '10px'}
            ),

            html.Label("Mutation Type:", style={'color': '#FFF'}),
            dcc.Dropdown(
                id='mutation-type',
                options=[
                    {'label': 'Gaussian Mutation', 'value': 'gaussian'},
                    {'label': 'Uniform Mutation', 'value': 'uniform'},
                    {'label': 'Creep Mutation', 'value': 'creep'}
                ],
                value='gaussian',
                style={'width': '100%', 'marginBottom': '20px'}
            ),

            html.Button(
                'Run Algorithm',
                id='run-button',
                style={
                    'width': '100%',
                    'backgroundColor': '#4CAF50',
                    'color': 'white',
                    'padding': '14px 20px',
                    'border': 'none',
                    'borderRadius': '4px'
                }
            ),
            
        ], style={'width': '30%', 'padding': '20px'}),
        
        # Graph
        html.Div([
            dcc.Graph(id='fitness-plot')
        ], style={'width': '70%'})
    ], style={'display': 'flex', 'flexDirection': 'row'})
], style={'backgroundColor': '#1E1E1E', 'padding': '20px'})

@callback(
    Output('fitness-plot', 'figure'),
    [Input('run-button', 'n_clicks')],
    [State('fitness-function', 'value'),
     State('pop-size', 'value'),
     State('x1-min', 'value'),
     State('x1-max', 'value'),
     State('x2-min', 'value'),
     State('x2-max', 'value'),
     State('parent-selection', 'value'),
     State('survival-selection', 'value'),
     State('crossover-type', 'value'),
     State('mutation-type', 'value')]
)

def update_graph(n_clicks, fitness_func, pop_size, x1_min, x1_max, x2_min, x2_max, 
                parent_sel, survival_sel, crossover_type, mutation_type):
    if n_clicks is None:
        return {}
        
    try:
        # Run evolutionary algorithm with parameters
        results = evolutionary_algorithm(
            fitness_func=fitness_func,
            pop_size=pop_size,
            x1_range=(x1_min, x1_max),
            x2_range=(x2_min, x2_max),
            parent_selection=parent_sel,
            survival_selection=survival_sel,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            generations=40,  # You might want to make these configurable
            runs=10
        )
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each run
        for i in range(10):  # 10 runs
            fig.add_trace(go.Scatter(
                y=results['run_data'][i],
                mode='lines',
                name=f'Run {i+1}',
                opacity=0.3
            ))
        
        # Add average trace
        fig.add_trace(go.Scatter(
            y=results['average'],
            mode='lines',
            name='Average',
            line=dict(color='white', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title='Fitness Over Generations',
            xaxis_title='Generation',
            yaxis_title='Fitness',
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='white',
                borderwidth=1
            )
        )
        
        return fig
    except Exception as e:
        print(f"Error: {str(e)}")  # For debugging
        return {}