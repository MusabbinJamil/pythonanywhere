import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
import numpy as np
from evo_algo import evolutionary_algorithm, get_results_csv_string
from gen_algo import genetic_algorithm
from ant_colony import ant_colony_optimization
from diff_evo import differential_evolution
from par_swarm import particle_swarm_optimization
import math

layout = html.Div([
    html.H1("Computational Intelligence", 
            style={'textAlign': 'center', 'color': '#FFF', 'marginBottom': '20px'}),
    
    html.Br(),

    # Tabs for different algorithms
    dcc.Tabs([
        # Tab for Evolutionary Algorithm
        dcc.Tab(label="Evolutionary Algorithm", children=[
            html.Br(),
            html.Div([                
                # Input fields for Evolutionary Algorithm
                html.Div([
                    html.Label("Fitness Function:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='evo-fitness-function',
                        type='text',
                        placeholder='e.g., x**2 + y**2',
                        value='x**2 + y**2',
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    
                    html.Label("Population Size:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='evo-pop-size',
                        type='number',
                        value=10,
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    
                    html.Label("(x1, y1): ", style={'color': '#FFF'}),
                    dcc.Input(
                        id='evo-x1-min',
                        type='number',
                        value=-5,
                        placeholder='Min',
                        style={'width': '20%', 'marginRight': '10%'}
                    ),
                    dcc.Input(
                        id='evo-x1-max',
                        type='number',
                        value=5,
                        placeholder='Max',
                        style={'width': '20%'}
                    ),

                    html.Br(),
                    
                    html.Label("(x2, y2): ", style={'color': '#FFF'}),
                    dcc.Input(
                        id='evo-x2-min',
                        type='number',
                        value=-5,
                        placeholder='Min',
                        style={'width': '20%', 'marginRight': '10%'}
                    ),
                    dcc.Input(
                        id='evo-x2-max',
                        type='number',
                        value=5,
                        placeholder='Max',
                        style={'width': '20%'}
                    ),

                    html.Br(),
                    
                    html.Label("Parent Selection:", style={'color': '#FFF'}),
                    dcc.Dropdown(
                        id='evo-parent-selection',
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
                        id='evo-survival-selection',
                        options=[
                            {'label': 'Truncation', 'value': 'truncation'},
                            {'label': 'Tournament', 'value': 'tournament'}
                        ],
                        value='truncation',
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),

                    html.Label("Crossover Type:", style={'color': '#FFF'}),
                    dcc.Dropdown(
                        id='evo-crossover-type',
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
                        id='evo-mutation-type',
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
                        id='evo-run-button',
                        style={
                            'width': '100%',
                            'backgroundColor': '#4CAF50',
                            'color': 'white',
                            'padding': '14px 20px',
                            'border': 'none',
                            'borderRadius': '4px'
                        }
                    ),

                    html.Button(
                        'Download CSV',
                        id='evo-download-csv-button',
                        style={
                            'width': '100%',
                            'backgroundColor': '#2196F3',
                            'color': 'white',
                            'padding': '14px 20px',
                            'marginTop': '10px',
                            'border': 'none',
                            'borderRadius': '4px'
                        }
                    ),
                    
                    dcc.Download(id='evo-download-csv'),
                    
                ], style={'width': '30%', 'padding': '20px'}),

                # Graph for Evolutionary Algorithm
                html.Div([
                    dcc.Loading(
                        id="evo-loading",
                        type="circle",
                        children=[
                            dcc.Graph(id='evo-fitness-plot')
                        ],
                        color="#00BFFF"
                    )
                ], style={'width': '70%'})
            ], style={'display': 'flex', 'flexDirection': 'row'}),
        
    ], style={'backgroundColor': '#1E1E1E'}),

        # Tab for Genetic Algorithm
        dcc.Tab(label="Genetic Algorithm", children=[
            html.Br(),
            html.Div([    
                # Input fields for Genetic Algorithm
                html.Div([
                    html.Label("Fitness Function:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='gen-fitness-function',
                        type='text',
                        placeholder='e.g., x1**2 + x2**2',
                        value='x1**2 + x2**2',
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    
                    html.Label("Population Size:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='gen-pop-size',
                        type='number',
                        value=50,
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    
                    html.Label("Chromosome Length:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='gen-chromosome-length',
                        type='number',
                        value=20,
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    
                    html.Label("X1 Range:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='gen-x1-min',
                        type='number',
                        value=-5,
                        placeholder='Min',
                        style={'width': '20%', 'marginRight': '10%'}
                    ),
                    dcc.Input(
                        id='gen-x1-max',
                        type='number',
                        value=5,
                        placeholder='Max',
                        style={'width': '20%'}
                    ),

                    html.Br(),
                    
                    html.Label("X2 Range:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='gen-x2-min',
                        type='number',
                        value=-5,
                        placeholder='Min',
                        style={'width': '20%', 'marginRight': '10%'}
                    ),
                    dcc.Input(
                        id='gen-x2-max',
                        type='number',
                        value=5,
                        placeholder='Max',
                        style={'width': '20%'}
                    ),

                    html.Br(),
                    
                    html.Label("Selection Method:", style={'color': '#FFF'}),
                    dcc.Dropdown(
                        id='gen-selection-method',
                        options=[
                            {'label': 'Tournament', 'value': 'tournament'},
                            {'label': 'Roulette Wheel', 'value': 'roulette'},
                            {'label': 'Rank Based', 'value': 'rank'}
                        ],
                        value='tournament',
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),

                    html.Label("Crossover Method:", style={'color': '#FFF'}),
                    dcc.Dropdown(
                        id='gen-crossover-method',
                        options=[
                            {'label': 'Single Point', 'value': 'single_point'},
                            {'label': 'Two Point', 'value': 'two_point'},
                            {'label': 'Uniform', 'value': 'uniform'}
                        ],
                        value='single_point',
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),

                    html.Label("Mutation Method:", style={'color': '#FFF'}),
                    dcc.Dropdown(
                        id='gen-mutation-method',
                        options=[
                            {'label': 'Bit Flip', 'value': 'bit_flip'},
                            {'label': 'Swap', 'value': 'swap'},
                            {'label': 'Inversion', 'value': 'inversion'}
                        ],
                        value='bit_flip',
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    
                    html.Label("Mutation Rate:", style={'color': '#FFF'}),
                    html.Div([
                        dcc.Slider(
                            id='gen-mutation-rate',
                            min=0.001,
                            max=0.1,
                            step=0.001,
                            value=0.01,
                            marks={0.001: '0.001', 0.05: '0.05', 0.1: '0.1'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'width': '100%', 'marginBottom': '20px', 'color': '#FFF'}),
                    
                    html.Label("Elite Size:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='gen-elite-size',
                        type='number',
                        value=2,
                        style={'width': '100%', 'marginBottom': '20px'}
                    ),

                    html.Button(
                        'Run Algorithm',
                        id='gen-run-button',
                        style={
                            'width': '100%',
                            'backgroundColor': '#4CAF50',
                            'color': 'white',
                            'padding': '14px 20px',
                            'border': 'none',
                            'borderRadius': '4px'
                        }
                    ),

                    html.Div([
                        html.Button(
                            'Download CSV',
                            id='gen-download-csv-button',
                            style={
                                'width': '100%',
                                'backgroundColor': '#2196F3',
                                'color': 'white',
                                'padding': '10px',
                                'marginTop': '10px',
                                'border': 'none',
                                'borderRadius': '4px'
                            }
                        ),
                        dcc.Download(id='gen-download-csv'),
                        # Add this store component
                        dcc.Store(id='gen-results-store'),
                    ], style={'marginTop': '10px'}),
                    
                ], style={'width': '30%', 'padding': '20px'}),
                
                html.Br(),
                # Graph for Genetic Algorithm
                html.Div([
                    dcc.Loading(
                        id="gen-loading",
                        type="circle",
                        children=[
                            dcc.Graph(id='gen-fitness-plot')
                        ],
                        color="#00BFFF"
                    )
                ], style={'width': '70%'})

            ], style={'display': 'flex', 'flexDirection': 'row'}),            
        ], style={'backgroundColor': '#1E1E1E'}),
        # New Tab for Ant Colony Optimization
        dcc.Tab(label="Ant Colony Optimization", children=[
            html.Br(),
            html.Div([
                # Input fields for ACO Algorithm
                html.Div([
                    html.Label("Problem Type:", style={'color': '#FFF'}),
                    dcc.Dropdown(
                        id='aco-problem-type',
                        options=[
                            {'label': 'Traveling Salesman Problem', 'value': 'tsp'},
                            {'label': 'Binary Optimization', 'value': 'binary'}
                        ],
                        value='tsp',
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    
                    html.Label("Problem Size:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='aco-problem-size',
                        type='number',
                        value=10,
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    
                    html.Label("Number of Ants:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='aco-num-ants',
                        type='number',
                        value=20,
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    
                    html.Label("Number of Iterations:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='aco-iterations',
                        type='number',
                        value=100,
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    
                    html.Label("Alpha (Pheromone Importance):", style={'color': '#FFF'}),
                    html.Div([
                        dcc.Slider(
                            id='aco-alpha',
                            min=0.1,
                            max=5.0,
                            step=0.1,
                            value=1.0,
                            marks={0.1: '0.1', 1: '1', 3: '3', 5: '5'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                        ], style={'width': '100%', 'margin-bottom': '10px', 'color': '#FFF'}),
                        
                    html.Label("Beta (Heuristic Importance):", style={'color': '#FFF'}),
                    html.Div([
                        dcc.Slider(
                            id='aco-beta',
                            min=0.1,
                            max=10.0,
                            step=0.1,
                            value=2.0,
                            marks={0.1: '0.1', 2: '2', 5: '5', 10: '10'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'width': '100%', 'margin-bottom': '10px', 'color': '#FFF'}),
                    
                    html.Label("Evaporation Rate:", style={'color': '#FFF'}),
                    html.Div([
                        dcc.Slider(
                            id='aco-evaporation',
                            min=0.01,
                            max=0.99,
                            step=0.01,
                            value=0.5,
                            marks={0.01: '0.01', 0.5: '0.5', 0.99: '0.99'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'width': '100%', 'margin-bottom': '10px', 'color': '#FFF'}),
                    
                    html.Br(),
                    
                    html.Label("Initial Pheromone:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='aco-init-pheromone',
                        type='number',
                        value=0.1,
                        step=0.01,
                        style={'width': '100%', 'marginBottom': '20px'}
                    ),

                    html.Label("Problem Definition:", style={'color': '#FFF'}),
                    dcc.RadioItems(
                        id='aco-problem-definition',
                        options=[
                            {'label': 'Random Generation', 'value': 'random'},
                            {'label': 'Custom Input', 'value': 'custom'}
                        ],
                        value='random',
                        style={'color': '#FFF', 'marginBottom': '10px'}
                    ),

                    html.Div(id='aco-custom-input-container', children=[
                        html.Label("City Coordinates (x,y pairs, one per line):", style={'color': '#FFF'}),
                        dcc.Textarea(
                            id='aco-custom-coordinates',
                            placeholder='0,0\n10,20\n30,40\n...',
                            style={'width': '100%', 'height': '100px', 'marginBottom': '10px'},
                            disabled=True
                        )
                    ], style={'display': 'none'}),

                    html.Div(id='aco-binary-fitness-container', children=[
                        html.Label("Binary Fitness Function:", style={'color': '#FFF'}),
                        dcc.Input(
                            id='aco-binary-fitness',
                            type='text',
                            placeholder='e.g., sum(solution) or more complex formula',
                            value='sum(solution)',
                            style={'width': '100%', 'marginBottom': '10px'}
                        )
                    ], style={'display': 'none'}),
                    
                    html.Button(
                        'Run Algorithm',
                        id='aco-run-button',
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
                
                # Graph container for ACO
                html.Div([
                    html.Div([
                        dcc.Loading(
                            id="aco-convergence-loading",
                            type="circle",
                            children=[
                                dcc.Graph(id='aco-convergence-plot')
                            ],
                            color="#00BFFF"
                        )
                    ], style={'height': '50%'}),
                    html.Div([
                        dcc.Loading(
                            id="aco-solution-loading",
                            type="circle",
                            children=[
                                dcc.Graph(id='aco-solution-plot')
                            ],
                            color="#00BFFF"
                        )
                    ], style={'height': '50%'})
                ], style={'width': '70%'})

                
            ], style={'display': 'flex', 'flexDirection': 'row'})
        ], style={'backgroundColor': '#1E1E1E'}),
        # Tab for Differential Evolution
        dcc.Tab(label="Differential Evolution", children=[
            html.Br(),
            html.Div([
                # Input fields for Differential Evolution
                html.Div([
                    html.Label("Fitness Function:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='de-fitness-function',
                        type='text',
                        placeholder='e.g., x**2 + y**2',
                        value='x**2 + y**2',
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    
                    html.Label("Population Size:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='de-pop-size',
                        type='number',
                        value=30,
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    
                    html.Label("X Range:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='de-x-min',
                        type='number',
                        value=-5,
                        placeholder='Min',
                        style={'width': '20%', 'marginRight': '10%'}
                    ),
                    dcc.Input(
                        id='de-x-max',
                        type='number',
                        value=5,
                        placeholder='Max',
                        style={'width': '20%'}
                    ),

                    html.Br(),
                    
                    html.Label("Y Range:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='de-y-min',
                        type='number',
                        value=-5,
                        placeholder='Min',
                        style={'width': '20%', 'marginRight': '10%'}
                    ),
                    dcc.Input(
                        id='de-y-max',
                        type='number',
                        value=5,
                        placeholder='Max',
                        style={'width': '20%'}
                    ),

                    html.Br(),
                    
                    html.Label("Strategy:", style={'color': '#FFF'}),
                    dcc.Dropdown(
                        id='de-strategy',
                        options=[
                            {'label': 'DE/rand/1/bin', 'value': 'rand1bin'},
                            {'label': 'DE/best/1/bin', 'value': 'best1bin'},
                            {'label': 'DE/rand/2/bin', 'value': 'rand2bin'},
                            {'label': 'DE/best/2/bin', 'value': 'best2bin'}
                        ],
                        value='rand1bin',
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    
                    html.Label("Mutation Factor (F):", style={'color': '#FFF'}),
                    html.Div([
                        dcc.Slider(
                            id='de-mutation-factor',
                            min=0.1,
                            max=2.0,
                            step=0.1,
                            value=0.8,
                            marks={0.1: '0.1', 0.8: '0.8', 1.5: '1.5', 2.0: '2.0'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'width': '100%', 'marginBottom': '10px', 'color': '#FFF'}),
                    
                    html.Label("Crossover Probability (CR):", style={'color': '#FFF'}),
                    html.Div([
                        dcc.Slider(
                            id='de-crossover-prob',
                            min=0.1,
                            max=1.0,
                            step=0.1,
                            value=0.7,
                            marks={0.1: '0.1', 0.5: '0.5', 0.9: '0.9'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'width': '100%', 'marginBottom': '20px', 'color': '#FFF'}),
                    
                    html.Label("Number of Generations:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='de-generations',
                        type='number',
                        value=100,
                        style={'width': '100%', 'marginBottom': '20px'}
                    ),

                    html.Button(
                        'Run Algorithm',
                        id='de-run-button',
                        style={
                            'width': '100%',
                            'backgroundColor': '#4CAF50',
                            'color': 'white',
                            'padding': '14px 20px',
                            'border': 'none',
                            'borderRadius': '4px'
                        }
                    ),

                    html.Button(
                        'Download CSV',
                        id='de-download-csv-button',
                        style={
                            'width': '100%',
                            'backgroundColor': '#2196F3',
                            'color': 'white',
                            'padding': '14px 20px',
                            'marginTop': '10px',
                            'border': 'none',
                            'borderRadius': '4px'
                        }
                    ),
                    
                    dcc.Download(id='de-download-csv'),
                    dcc.Store(id='de-results-store'),
                    
                ], style={'width': '30%', 'padding': '20px'}),

                # Graph for Differential Evolution
                html.Div([
                    dcc.Loading(
                        id="de-loading",
                        type="circle",
                        children=[
                            dcc.Graph(id='de-fitness-plot')
                        ],
                        color="#00BFFF"
                    )
                ], style={'width': '70%'})
            ], style={'display': 'flex', 'flexDirection': 'row'}),
        ], style={'backgroundColor': '#1E1E1E'}),

        # Tab for Particle Swarm Optimization
        dcc.Tab(label="Particle Swarm Optimization", children=[
            html.Br(),
            html.Div([
                # Input fields for PSO
                html.Div([
                    html.Label("Fitness Function:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='pso-fitness-function',
                        type='text',
                        placeholder='e.g., x**2 + y**2',
                        value='x**2 + y**2',
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    
                    html.Label("Number of Particles:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='pso-num-particles',
                        type='number',
                        value=30,
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    
                    html.Label("X Range:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='pso-x-min',
                        type='number',
                        value=-5,
                        placeholder='Min',
                        style={'width': '20%', 'marginRight': '10%'}
                    ),
                    dcc.Input(
                        id='pso-x-max',
                        type='number',
                        value=5,
                        placeholder='Max',
                        style={'width': '20%'}
                    ),

                    html.Br(),
                    
                    html.Label("Y Range:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='pso-y-min',
                        type='number',
                        value=-5,
                        placeholder='Min',
                        style={'width': '20%', 'marginRight': '10%'}
                    ),
                    dcc.Input(
                        id='pso-y-max',
                        type='number',
                        value=5,
                        placeholder='Max',
                        style={'width': '20%'}
                    ),

                    html.Br(),
                    
                    html.Label("Inertia Weight (w):", style={'color': '#FFF'}),
                    html.Div([
                        dcc.Slider(
                            id='pso-inertia',
                            min=0.1,
                            max=1.0,
                            step=0.05,
                            value=0.5,
                            marks={0.1: '0.1', 0.5: '0.5', 1.0: '1.0'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'width': '100%', 'marginBottom': '10px', 'color': '#FFF'}),
                    
                    html.Label("Cognitive Coefficient (c1):", style={'color': '#FFF'}),
                    html.Div([
                        dcc.Slider(
                            id='pso-c1',
                            min=0.5,
                            max=2.5,
                            step=0.1,
                            value=1.5,
                            marks={0.5: '0.5', 1.5: '1.5', 2.5: '2.5'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'width': '100%', 'marginBottom': '10px', 'color': '#FFF'}),
                    
                    html.Label("Social Coefficient (c2):", style={'color': '#FFF'}),
                    html.Div([
                        dcc.Slider(
                            id='pso-c2',
                            min=0.5,
                            max=2.5,
                            step=0.1,
                            value=1.5,
                            marks={0.5: '0.5', 1.5: '1.5', 2.5: '2.5'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'width': '100%', 'marginBottom': '20px', 'color': '#FFF'}),
                    
                    html.Label("Number of Iterations:", style={'color': '#FFF'}),
                    dcc.Input(
                        id='pso-iterations',
                        type='number',
                        value=100,
                        style={'width': '100%', 'marginBottom': '20px'}
                    ),

                    html.Button(
                        'Run Algorithm',
                        id='pso-run-button',
                        style={
                            'width': '100%',
                            'backgroundColor': '#4CAF50',
                            'color': 'white',
                            'padding': '14px 20px',
                            'border': 'none',
                            'borderRadius': '4px'
                        }
                    ),

                    html.Button(
                        'Download CSV',
                        id='pso-download-csv-button',
                        style={
                            'width': '100%',
                            'backgroundColor': '#2196F3',
                            'color': 'white',
                            'padding': '14px 20px',
                            'marginTop': '10px',
                            'border': 'none',
                            'borderRadius': '4px'
                        }
                    ),
                    
                    dcc.Download(id='pso-download-csv'),
                    dcc.Store(id='pso-results-store'),
                    
                ], style={'width': '30%', 'padding': '20px'}),

                # Graph for Particle Swarm Optimization
                html.Div([
                    dcc.Loading(
                        id="pso-loading",
                        type="circle",
                        children=[
                            dcc.Graph(id='pso-fitness-plot')
                        ],
                        color="#00BFFF"
                    )
                ], style={'width': '70%'})
            ], style={'display': 'flex', 'flexDirection': 'row'}),
        ], style={'backgroundColor': '#1E1E1E'})
    ], style={'color': '#FFF'})
], style={'backgroundColor': '#1E1E1E', 'padding': '20px'})

#  Callback for Evolutionary Algorithm
@callback(
    Output('evo-fitness-plot', 'figure'),  # Remove the square brackets
    [Input('evo-run-button', 'n_clicks')],
    [State('evo-fitness-function', 'value'),
     State('evo-pop-size', 'value'),
     State('evo-x1-min', 'value'),
     State('evo-x1-max', 'value'),
     State('evo-x2-min', 'value'),
     State('evo-x2-max', 'value'),
     State('evo-parent-selection', 'value'),
     State('evo-survival-selection', 'value'),
     State('evo-crossover-type', 'value'),
     State('evo-mutation-type', 'value')]
)
def update_evo_graph(n_clicks, fitness_func, pop_size, x1_min, x1_max, x2_min, x2_max, 
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
            generations=40,
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
            title='Evolutionary Algorithm: Fitness Over Generations',
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
        return {}
    
@callback(
    Output('evo-download-csv', 'data'),
    [Input('evo-download-csv-button', 'n_clicks')],
    [State('evo-fitness-function', 'value')]
)
def download_evo_csv(n_clicks, fitness_func):
    if n_clicks is None:
        return None
        
    # Get the results from dcc.Store or recalculate
    try:
        # For simplicity, we're recreating the results
        # In a real app, you might want to store results in dcc.Store after calculation
        results = evolutionary_algorithm(
            fitness_func=fitness_func,
            pop_size=10,  # Using default values
            generations=40,
            runs=10
        )
        
        # Get CSV data as string
        csv_string = get_results_csv_string(results)
        
        return dict(
            content=csv_string,
            filename=f"evo_algo_results_{fitness_func.replace('**', 'pow').replace('*', 'mul')}.csv",
            type='text/csv'
        )
    except Exception as e:
        print(f"Error generating CSV: {str(e)}")
        return None

# Callback for Genetic Algorithm
@callback(
    [Output('gen-fitness-plot', 'figure'), 
     Output('gen-results-store', 'data')],  # Add output for store
    [Input('gen-run-button', 'n_clicks')],
    [State('gen-fitness-function', 'value'),
     State('gen-pop-size', 'value'),
     State('gen-chromosome-length', 'value'),
     State('gen-x1-min', 'value'),
     State('gen-x1-max', 'value'),
     State('gen-x2-min', 'value'),
     State('gen-x2-max', 'value'),
     State('gen-selection-method', 'value'),
     State('gen-crossover-method', 'value'),
     State('gen-mutation-method', 'value'),
     State('gen-mutation-rate', 'value'),
     State('gen-elite-size', 'value')]
)
def update_gen_graph(n_clicks, fitness_func, pop_size, chromosome_length, 
                    x1_min, x1_max, x2_min, x2_max, selection_method,
                    crossover_method, mutation_method, mutation_rate, elite_size):
    if n_clicks is None:
        return {}, None
        
    try:
        # Run genetic algorithm with parameters
        results = genetic_algorithm(
            fitness_func=fitness_func,
            chromosome_length=chromosome_length,
            pop_size=pop_size,
            variable_ranges=[(x1_min, x1_max), (x2_min, x2_max)],
            n_variables=2,
            selection_method=selection_method,
            crossover_method=crossover_method,
            mutation_method=mutation_method,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            generations=100,
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
            title='Genetic Algorithm: Fitness Over Generations',
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
        
        # Store fitness function with the results
        stored_data = {
            'results': results,
            'fitness_func': fitness_func
        }
        
        return fig, stored_data
    except Exception as e:
        return {}, None
    
@callback(
    Output('gen-download-csv', 'data'),
    [Input('gen-download-csv-button', 'n_clicks')],
    [State('gen-results-store', 'data')]
)
def download_gen_csv(n_clicks, stored_data):
    if n_clicks is None or stored_data is None:
        return None
        
    try:
        # Get results from the store
        results = stored_data['results']
        fitness_func = stored_data['fitness_func']
        
        # Get CSV data as string
        csv_string = get_results_csv_string(results)
        
        # Create sanitized filename from the fitness function
        safe_filename = fitness_func.replace('**', 'pow').replace('*', 'mul').replace('/', 'div')
        
        return dict(
            content=csv_string,
            filename=f"genetic_algo_results_{safe_filename}.csv",
            type='text/csv'
        )
    except Exception as e:
        print(f"Error generating CSV: {str(e)}")
        return None
    
@callback(
    [Output('aco-convergence-plot', 'figure'),
     Output('aco-solution-plot', 'figure')],
    [Input('aco-run-button', 'n_clicks')],
    [State('aco-problem-type', 'value'),
     State('aco-problem-size', 'value'),
     State('aco-num-ants', 'value'),
     State('aco-iterations', 'value'),
     State('aco-alpha', 'value'),
     State('aco-beta', 'value'),
     State('aco-evaporation', 'value'),
     State('aco-init-pheromone', 'value'),
     State('aco-problem-definition', 'value'),
     State('aco-custom-coordinates', 'value'),
     State('aco-binary-fitness', 'value')]
)
def update_aco_graph(n_clicks, problem_type, problem_size, num_ants, 
                    iterations, alpha, beta, evaporation, init_pheromone,
                    problem_definition, custom_coordinates, binary_fitness):

    print(f"ACO callback triggered with n_clicks: {n_clicks}")

    if n_clicks is None:
        return {}, {}
        
    try:
        print(f"Running ACO with problem_type: {problem_type}, size: {problem_size}")
        # Process custom coordinates if provided
        parsed_coordinates = None
        if problem_definition == 'custom' and custom_coordinates:
            try:
                # Parse the custom coordinates from text
                parsed_coordinates = []
                for line in custom_coordinates.strip().split('\n'):
                    if line.strip():
                        x, y = map(float, line.strip().split(','))
                        parsed_coordinates.append((x, y))
            except Exception as e:
                print(f"Error parsing coordinates: {e}")
        
        # Run ant colony optimization with parameters
        results = ant_colony_optimization(
            problem_type=problem_type,
            problem_size=problem_size,
            num_ants=num_ants,
            iterations=iterations,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evaporation,
            initial_pheromone=init_pheromone,
            runs=10,
            custom_coordinates=parsed_coordinates,
            custom_binary_fitness=binary_fitness if problem_type == 'binary' else None
        )
        
        # Create convergence figure
        convergence_fig = go.Figure()
        
        # Add traces for each run
        for i in range(10):  # 10 runs
            convergence_fig.add_trace(go.Scatter(
                y=results['run_data'][i],
                mode='lines',
                name=f'Run {i+1}',
                opacity=0.3
            ))
        
        # Add average trace
        convergence_fig.add_trace(go.Scatter(
            y=results['average'],
            mode='lines',
            name='Average',
            line=dict(color='white', width=3, dash='dash')
        ))
        
        convergence_fig.update_layout(
            title='ACO: Best Solution Quality Over Iterations',
            xaxis_title='Iteration',
            yaxis_title='Solution Quality',
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
        
        # Create solution visualization figure
        solution_fig = go.Figure()
        
        # For TSP problems, show the best route
        if problem_type == 'tsp':
            best_solution = results['best_solution']
            coordinates = results['coordinates']
            best_quality = results['best_quality']
            
            # Connect the points in the optimal tour
            x_coords = [coordinates[i][0] for i in best_solution] + [coordinates[best_solution[0]][0]]
            y_coords = [coordinates[i][1] for i in best_solution] + [coordinates[best_solution[0]][1]]
            
            solution_fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines+markers',
                name='Best Tour',
                line=dict(color='cyan', width=2),
                marker=dict(size=10)
            ))
            
            # Add city labels with visit order
            for idx, city_idx in enumerate(best_solution):
                solution_fig.add_annotation(
                    x=coordinates[city_idx][0],
                    y=coordinates[city_idx][1],
                    text=f"{city_idx}[{idx}]",
                    showarrow=False,
                    font=dict(color='white', size=12)
                )
            
            # Add distance labels on edges
            for i in range(len(best_solution)):
                next_i = (i + 1) % len(best_solution)
                city1 = best_solution[i]
                city2 = best_solution[next_i]
                
                # Calculate midpoint for the edge label
                x1, y1 = coordinates[city1]
                x2, y2 = coordinates[city2]
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                # Calculate distance between these cities
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                solution_fig.add_annotation(
                    x=mid_x,
                    y=mid_y,
                    text=f"{distance:.2f}",
                    showarrow=False,
                    font=dict(color='yellow', size=10),
                    bgcolor='rgba(0,0,0,0.5)'
                )
            
            # Add total distance annotation
            solution_fig.add_annotation(
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                text=f"Total Distance: {best_quality:.2f}",
                showarrow=False,
                font=dict(color='white', size=14),
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='cyan',
                borderwidth=2
            )
            
            # Add path order as text
            path_order_text = "Path: " + " → ".join([str(city) for city in best_solution]) + f" → {best_solution[0]}"
            solution_fig.add_annotation(
                x=0.5,
                y=-0.15,
                xref="paper",
                yref="paper",
                text=path_order_text,
                showarrow=False,
                font=dict(color='white', size=12),
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='cyan',
                borderwidth=1
            )
            
            solution_fig.update_layout(
                title='ACO: Best Tour Found',
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font=dict(color='white')
            )
        
        return convergence_fig, solution_fig
    
    except Exception as e:
        empty_fig = {
            'data': [],
            'layout': {
                'title': f'Error: {str(e)}',
                'plot_bgcolor': '#1E1E1E',
                'paper_bgcolor': '#1E1E1E',
                'font': {'color': 'white'}
            }
        }
        return empty_fig, empty_fig
    
@callback(
    [Output('aco-custom-input-container', 'style'),
     Output('aco-binary-fitness-container', 'style'),
     Output('aco-custom-coordinates', 'disabled')],
    [Input('aco-problem-definition', 'value'),
     Input('aco-problem-type', 'value')]
)
def update_problem_input_visibility(definition_type, problem_type):
    custom_coords_style = {'display': 'block'} if definition_type == 'custom' and problem_type == 'tsp' else {'display': 'none'}
    binary_fitness_style = {'display': 'block'} if problem_type == 'binary' else {'display': 'none'}
    coords_disabled = definition_type != 'custom'
    
    return custom_coords_style, binary_fitness_style, coords_disabled


# Callback for Differential Evolution
@callback(
    [Output('de-fitness-plot', 'figure'), 
     Output('de-results-store', 'data')],
    [Input('de-run-button', 'n_clicks')],
    [State('de-fitness-function', 'value'),
     State('de-pop-size', 'value'),
     State('de-x-min', 'value'),
     State('de-x-max', 'value'),
     State('de-y-min', 'value'),
     State('de-y-max', 'value'),
     State('de-strategy', 'value'),
     State('de-mutation-factor', 'value'),
     State('de-crossover-prob', 'value'),
     State('de-generations', 'value')]
)
def update_de_graph(n_clicks, fitness_func, pop_size, x_min, x_max, y_min, y_max, 
                   strategy, mutation_factor, crossover_prob, generations):
    if n_clicks is None:
        return {}, None
        
    try:
        # Run differential evolution algorithm with parameters
        results = differential_evolution(
            fitness_func=fitness_func,
            pop_size=pop_size,
            bounds=[(x_min, x_max), (y_min, y_max)],
            strategy=strategy,
            mutation_factor=mutation_factor,
            crossover_prob=crossover_prob,
            generations=generations,
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
            title='Differential Evolution: Fitness Over Generations',
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
        
        # Store results and fitness function
        stored_data = {
            'results': results,
            'fitness_func': fitness_func
        }
        
        return fig, stored_data
    except Exception as e:
        empty_fig = {
            'data': [],
            'layout': {
                'title': f'Error: {str(e)}',
                'plot_bgcolor': '#1E1E1E',
                'paper_bgcolor': '#1E1E1E',
                'font': {'color': 'white'}
            }
        }
        return empty_fig, None

@callback(
    Output('de-download-csv', 'data'),
    [Input('de-download-csv-button', 'n_clicks')],
    [State('de-results-store', 'data')]
)
def download_de_csv(n_clicks, stored_data):
    if n_clicks is None or stored_data is None:
        return None
        
    try:
        # Get results from the store
        results = stored_data['results']
        fitness_func = stored_data['fitness_func']
        
        # Get CSV data as string
        csv_string = get_results_csv_string(results)
        
        # Create sanitized filename from the fitness function
        safe_filename = fitness_func.replace('**', 'pow').replace('*', 'mul').replace('/', 'div')
        
        return dict(
            content=csv_string,
            filename=f"diff_evo_results_{safe_filename}.csv",
            type='text/csv'
        )
    except Exception as e:
        print(f"Error generating CSV: {str(e)}")
        return None

# Callback for Particle Swarm Optimization
@callback(
    [Output('pso-fitness-plot', 'figure'), 
     Output('pso-results-store', 'data')],
    [Input('pso-run-button', 'n_clicks')],
    [State('pso-fitness-function', 'value'),
     State('pso-num-particles', 'value'),
     State('pso-x-min', 'value'),
     State('pso-x-max', 'value'),
     State('pso-y-min', 'value'),
     State('pso-y-max', 'value'),
     State('pso-inertia', 'value'),
     State('pso-c1', 'value'),
     State('pso-c2', 'value'),
     State('pso-iterations', 'value')]
)
def update_pso_graph(n_clicks, fitness_func, num_particles, x_min, x_max, y_min, y_max, 
                   inertia, c1, c2, iterations):
    if n_clicks is None:
        return {}, None
        
    try:
        # Run particle swarm optimization with parameters
        results = particle_swarm_optimization(
            fitness_func=fitness_func,
            num_particles=num_particles,
            bounds=[(x_min, x_max), (y_min, y_max)],
            w=inertia,
            c1=c1,
            c2=c2,
            iterations=iterations,
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
            title='Particle Swarm Optimization: Fitness Over Iterations',
            xaxis_title='Iteration',
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
        
        # Store results and fitness function
        stored_data = {
            'results': results,
            'fitness_func': fitness_func
        }
        
        return fig, stored_data
    except Exception as e:
        empty_fig = {
            'data': [],
            'layout': {
                'title': f'Error: {str(e)}',
                'plot_bgcolor': '#1E1E1E',
                'paper_bgcolor': '#1E1E1E',
                'font': {'color': 'white'}
            }
        }
        return empty_fig, None

@callback(
    Output('pso-download-csv', 'data'),
    [Input('pso-download-csv-button', 'n_clicks')],
    [State('pso-results-store', 'data')]
)
def download_pso_csv(n_clicks, stored_data):
    if n_clicks is None or stored_data is None:
        return None
        
    try:
        # Get results from the store
        results = stored_data['results']
        fitness_func = stored_data['fitness_func']
        
        # Get CSV data as string
        csv_string = get_results_csv_string(results)
        
        # Create sanitized filename from the fitness function
        safe_filename = fitness_func.replace('**', 'pow').replace('*', 'mul').replace('/', 'div')
        
        return dict(
            content=csv_string,
            filename=f"pso_results_{safe_filename}.csv",
            type='text/csv'
        )
    except Exception as e:
        print(f"Error generating CSV: {str(e)}")
        return None

@callback(
    Output('bn-output', 'children', allow_duplicate=True),
    [Input('evo-run-button', 'n_clicks'),
     Input('gen-run-button', 'n_clicks'),
     Input('aco-run-button', 'n_clicks'),
     Input('de-run-button', 'n_clicks'),
     Input('pso-run-button', 'n_clicks')],
    prevent_initial_call=True
)
def run_algorithm(*args):
    ctx = dash.ctx  # <-- Use dash.ctx instead of dash.callback_context

    if not ctx.triggered:
        print("Callback triggered but no trigger information found")
        return dash.no_update

    # Get the ID of the button that was clicked
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"Button clicked: {button_id}")

    result = None
    if button_id == 'evo-run-button':
        result = "Running Evolutionary Algorithm..."
        print("Evolutionary Algorithm button clicked")
    elif button_id == 'gen-run-button':
        result = "Running Genetic Algorithm..."
        print("Genetic Algorithm button clicked")
    elif button_id == 'aco-run-button':
        result = "Running Ant Colony Optimization..."
        print("Ant Colony Optimization button clicked")
    elif button_id == 'de-run-button':
        result = "Running Differential Evolution..."
        print("Differential Evolution button clicked")
    elif button_id == 'pso-run-button':
        result = "Running Particle Swarm Optimization..."
        print("Particle Swarm Optimization button clicked")
    else:
        result = dash.no_update

    print(f"Returning result: {result}")
    return result
