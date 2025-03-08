<<<<<<< HEAD
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
=======
import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
import numpy as np
from evo_algo import evolutionary_algorithm, get_results_csv_string
from gen_algo import genetic_algorithm
from ant_colony import ant_colony_optimization

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

    if n_clicks is None:
        return {}, {}
        
    try:
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
            
            # Add city labels
            for i, coord in enumerate(coordinates):
                solution_fig.add_annotation(
                    x=coord[0],
                    y=coord[1],
                    text=str(i),
                    showarrow=False,
                    font=dict(color='white')
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
>>>>>>> master
