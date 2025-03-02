import dash
from dash import html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import re
import yaml

# Add pgmpy imports
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Global variable to store the Bayesian network model between callbacks
global_bn_model = None

# Define a function that returns the layout with tabs
def get_layout():
    return html.Div([
        html.H1("Probabilistic Reasoning", style={'textAlign': 'center'}),
        
        dcc.Tabs([
            # Tab 1: Graph Plotting
            dcc.Tab(label="Graph Plotting", style={'backgroundColor': '#3C3C3C', 'color': 'white'}, 
                   selected_style={'backgroundColor': '#4C4C4C', 'color': 'white'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Graph Structure Input"),
                        html.P("Enter graph structure in format: node: [connected_nodes]"),
                        html.P("Example:"),
                        html.Pre("""a: [b,c]
                                    b: [a,d]
                                    c: [a,d]
                                    d: [b,c]""", 
                            style={
                                'backgroundColor': '#2C2C2C',
                                'padding': '10px',
                                'borderRadius': '5px',
                                'color': '#CCCCCC'
                            }
                        ),
                        dcc.Textarea(
                            id='graph-structure-input',
                            placeholder='Enter graph structure...',
                            style={
                                'width': '100%',
                                'height': '200px',
                                'backgroundColor': '#3C3C3C',
                                'color': 'white',
                                'borderRadius': '5px',
                                'padding': '10px'
                            },
                            value='a: [b,c]\nb: [a,d]\nc: [a,d]\nd: [b,c]'
                        ),
                        html.Button('Plot Graph', id='plot-graph', style={
                            'backgroundColor': '#3C3C3C',
                            'color': 'white',
                            'border': '1px solid #FFF',
                            'padding': '10px',
                            'margin': '10px'
                        }),
                    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
                    
                    html.Div([
                        html.H3("Graph Algorithms"),
                        html.P("Select an algorithm to apply to the graph:"),
                        dcc.Dropdown(
                            id='graph-algorithm',
                            options=[
                                {'label': 'None (Basic Graph)', 'value': 'none'},
                                {'label': 'Maximum Cardinality Search', 'value': 'mcs'},
                                {'label': 'Variable Elimination', 'value': 've'},
                                {'label': 'Junction Tree', 'value': 'jt'},  # Add this new option
                            ],
                            value='none',
                            style={'backgroundColor': '#3C3C3C', 'color': 'black'}
                        ),
                        html.Button('Apply Algorithm', id='apply-algorithm', style={
                            'backgroundColor': '#3C3C3C',
                            'color': 'white',
                            'border': '1px solid #FFF',
                            'padding': '10px',
                            'margin': '10px'
                        }),
                    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
                ]),
                
                html.Div([
                    dcc.Graph(id='graph-display', style={'height': '600px'}),
                ]),
                
                html.Div(id='output-container', style={'margin': '20px', 'padding': '10px', 'border': '1px solid #ddd'})
            ]),
            
            # Tab 2: Bayesian Networks
            dcc.Tab(label="Bayesian Networks", style={'backgroundColor': '#3C3C3C', 'color': 'white'}, 
                   selected_style={'backgroundColor': '#4C4C4C', 'color': 'white'}, children=[
                html.Div([
                    html.Div([
                        html.H3("Bayesian Network Structure"),
                        html.P("Define nodes and their parents (format: child: [parent1,parent2,...]"),
                        dcc.Textarea(
                            id='bn-structure-input',
                            placeholder='Enter Bayesian network structure...',
                            style={
                                'width': '100%',
                                'height': '200px',
                                'backgroundColor': '#3C3C3C',
                                'color': 'white',
                                'borderRadius': '5px',
                                'padding': '10px'
                            },
                            value='rain: []\nsprinkler: [rain]\nwet_grass: [rain,sprinkler]'
                        ),
                    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
                    
                    html.Div([
                        html.H3("Conditional Probabilities"),
                        html.P("Define conditional probability tables:"),
                        html.Pre("""rain:
                                    true: 0.2
                                    false: 0.8
                                    sprinkler|rain:
                                    true|true: 0.01
                                    true|false: 0.4
                                    false|true: 0.99
                                    false|false: 0.6
                                    wet_grass|rain,sprinkler:
                                    true|true,true: 0.99
                                    true|true,false: 0.8
                                    true|false,true: 0.9
                                    true|false,false: 0.0
                                    false|true,true: 0.01
                                    false|true,false: 0.2
                                    false|false,true: 0.1
                                    false|false,false: 1.0""", 
                            style={
                                'backgroundColor': '#2C2C2C',
                                'padding': '10px',
                                'borderRadius': '5px',
                                'color': '#CCCCCC',
                                'fontSize': '12px'
                            }
                        ),
                        dcc.Textarea(
                            id='bn-cpt-input',
                            placeholder='Enter probability tables...',
                            style={
                                'width': '100%',
                                'height': '200px',
                                'backgroundColor': '#3C3C3C',
                                'color': 'white',
                                'borderRadius': '5px',
                                'padding': '10px'
                            },
                            value="""rain:\n  true: 0.2\n  false: 0.8\nsprinkler|rain:\n  true|true: 0.01\n  true|false: 0.4\n  false|true: 0.99\n  false|false: 0.6\nwet_grass|rain,sprinkler:\n  true|true,true: 0.99\n  true|true,false: 0.8\n  true|false,true: 0.9\n  true|false,false: 0.0\n  false|true,true: 0.01\n  false|true,false: 0.2\n  false|false,true: 0.1\n  false|false,false: 1.0"""
                        ),
                        html.Button('Create Bayesian Network', id='create-bn', style={
                            'backgroundColor': '#3C3C3C',
                            'color': 'white',
                            'border': '1px solid #FFF',
                            'padding': '10px',
                            'margin': '10px'
                        }),
                    ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
                ]),
                
                html.Div([
                    html.H3("Inference"),
                    html.Div([
                        html.Div([
                            html.P("Query (what to compute):"),
                            dcc.Input(
                                id='bn-query',
                                type='text',
                                placeholder='e.g., wet_grass=true',
                                style={
                                    'width': '100%',
                                    'backgroundColor': '#3C3C3C',
                                    'color': 'white',
                                    'borderRadius': '5px',
                                    'padding': '10px'
                                }
                            ),
                        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                        html.Div([
                            html.P("Evidence (what we know):"),
                            dcc.Input(
                                id='bn-evidence',
                                type='text',
                                placeholder='e.g., rain=true',
                                style={
                                    'width': '100%',
                                    'backgroundColor': '#3C3C3C',
                                    'color': 'white',
                                    'borderRadius': '5px',
                                    'padding': '10px'
                                }
                            ),
                        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
                    ]),
                    html.Button('Run Inference', id='run-inference', style={
                        'backgroundColor': '#3C3C3C',
                        'color': 'white',
                        'border': '1px solid #FFF',
                        'padding': '10px',
                        'margin': '10px'
                    }),
                ], style={'padding': '20px'}),
                
                html.Div([
                    dcc.Graph(id='bn-display', style={'height': '400px'}),
                ]),
                
                html.Div(id='bn-output', style={'margin': '20px', 'padding': '10px', 'border': '1px solid #ddd'})
            ]),
        ]),
    ])

# Keep the original layout variable for backward compatibility
layout = get_layout()

# Existing callback for graph plotting - no change needed
@callback(
    Output('graph-display', 'figure', allow_duplicate=True),
    Output('output-container', 'children', allow_duplicate=True),
    Input('apply-algorithm', 'n_clicks'),
    State('graph-structure-input', 'value'),
    State('graph-algorithm', 'value'),
    prevent_initial_call=True
)
def apply_graph_algorithm(n_clicks, input_text, algorithm):
    if not input_text:
        return go.Figure(), "Please enter graph structure."
    
    try:
        # Parse input text to create graph
        G = nx.Graph()
        lines = input_text.strip().split('\n')
        
        for line in lines:
            match = re.match(r'(\w+):\s*\[(.*?)\]', line)
            if match:
                node = match.group(1)
                neighbors = [n.strip() for n in match.group(2).split(',') if n.strip()]
                
                if node not in G.nodes():
                    G.add_node(node)
                
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)
        
        if not G.nodes():
            return go.Figure(), "No valid graph structure found."
        
        # Apply selected algorithm
        if algorithm == 'none':
            result_graph = G
            result_message = "Basic graph visualization."
            
        elif algorithm == 'mcs':
            # Maximum Cardinality Search
            ordering = list(nx.algorithms.ordering.maximum_cardinality_search(G))
            result_graph = G.copy()
            nx.set_node_attributes(result_graph, {node: {'order': idx} for idx, node in enumerate(ordering)})
            result_message = f"Maximum Cardinality Search ordering: {', '.join(ordering)}"
            
        elif algorithm == 've':
            # Variable Elimination
            try:
                ordering = list(nx.algorithms.approximation.treewidth_min_degree(G)[1])
                result_graph = G.copy()
                nx.set_node_attributes(result_graph, {node: {'order': idx} for idx, node in enumerate(ordering)})
                result_message = f"Variable Elimination ordering: {', '.join(ordering)}"
            except Exception as e:
                result_graph = G
                result_message = f"Error applying Variable Elimination: {str(e)}"
        
        elif algorithm == 'jt':
            # Junction Tree algorithm
            try:
                # Step 1: Create a chordal graph (triangulate)
                try:
                    ordering = list(nx.algorithms.approximation.treewidth_min_degree(G)[1])
                except:
                    ordering = list(G.nodes())  # Fallback
                
                # Create triangulated graph
                triangulated_G = G.copy()
                for i, node in enumerate(ordering):
                    neighbors = list(triangulated_G.neighbors(node))
                    later_neighbors = [n for n in neighbors if n in ordering[i+1:]]
                    
                    # Connect all later neighbors (create a clique)
                    for u in later_neighbors:
                        for v in later_neighbors:
                            if u != v and not triangulated_G.has_edge(u, v):
                                triangulated_G.add_edge(u, v)
                                
                # Step 2: Find maximal cliques
                try:
                    cliques = list(nx.find_cliques(triangulated_G))
                except:
                    # Fallback approach
                    cliques = []
                    for node in triangulated_G.nodes():
                        cliques.append([node] + list(triangulated_G.neighbors(node)))
                
                # Step 3: Build junction tree
                junction_tree = nx.Graph()
                
                # Add nodes for each clique
                for i, clique in enumerate(cliques):
                    junction_tree.add_node(i, members=set(clique))
                
                # Connect cliques with edges weighted by intersection size
                for i in range(len(cliques)):
                    for j in range(i+1, len(cliques)):
                        intersection = set(cliques[i]).intersection(set(cliques[j]))
                        if intersection:
                            junction_tree.add_edge(i, j, weight=len(intersection), separator=intersection)
                
                # Create maximum spanning tree
                if len(junction_tree.edges()) > 0:
                    mst_edges = nx.maximum_spanning_tree(junction_tree, weight='weight').edges()
                    edges_to_remove = list(set(junction_tree.edges()) - set(mst_edges))
                    junction_tree.remove_edges_from(edges_to_remove)
                
                result_graph = junction_tree
                
                # Format results for display
                clique_info = []
                for i, clique in enumerate(cliques):
                    if i in result_graph.nodes():
                        clique_info.append(f"C{i}: {{{', '.join(clique)}}}")
                
                separator_info = []
                for u, v, data in result_graph.edges(data=True):
                    if 'separator' in data:
                        separator_info.append(f"S{u}-{v}: {{{', '.join(data['separator'])}}}")
                
                result_message = "Junction Tree created.\n\n"
                result_message += "Cliques:\n" + "\n".join(clique_info)
                result_message += "\n\nSeparators:\n" + "\n".join(separator_info)
                
            except Exception as e:
                result_graph = G
                result_message = f"Error creating Junction Tree: {str(e)}"
        else:
            result_graph = G
            result_message = "Unknown algorithm selected."
        
        # Visualize the resulting graph
        if algorithm == 'jt':
            # Special visualization for junction tree
            try:
                pos = nx.spring_layout(result_graph)
                
                # Create edges
                edge_trace = go.Scatter(
                    x=[], y=[], line=dict(width=2, color='#888'),
                    hoverinfo='text', mode='lines'
                )
                
                edge_labels = []
                for edge in result_graph.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_trace['x'] += (x0, x1, None)
                    edge_trace['y'] += (y0, y1, None)
                    
                    # Add separator label
                    if 'separator' in result_graph.edges[edge]:
                        separator = result_graph.edges[edge]['separator']
                        edge_labels.append({
                            'x': (x0 + x1) / 2,
                            'y': (y0 + y1) / 2,
                            'text': f"{{{', '.join(separator)}}}",
                            'showarrow': False,
                            'font': {'color': 'yellow'}
                        })
                
                # Create nodes (cliques)
                node_trace = go.Scatter(
                    x=[], y=[], text=[], mode='markers+text',
                    textposition="top center", textfont=dict(color='white'),
                    hoverinfo='text',
                    marker=dict(showscale=False, color='lightgreen', size=40,
                               line=dict(width=2, color='DarkSlateGrey'))
                )
                
                for node in result_graph.nodes():
                    x, y = pos[node]
                    node_trace['x'] += (x,)
                    node_trace['y'] += (y,)
                    
                    # Format node text as clique members
                    if 'members' in result_graph.nodes[node]:
                        members = result_graph.nodes[node]['members']
                        node_trace['text'] += (f"C{node}:\n{{{', '.join(members)}}}", )
                    else:
                        node_trace['text'] += (f"C{node}", )
                
                fig = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False, hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=edge_labels,
                        plot_bgcolor='#1E1E1E', paper_bgcolor='#1E1E1E',
                        font=dict(color='white'),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                )
            except Exception as e:
                # Fallback visualization
                fig = create_standard_graph_figure(G)
                result_message += f"\n\nError in junction tree visualization: {str(e)}"
        else:
            # Standard graph visualization with algorithm-specific coloring
            fig = create_standard_graph_figure(result_graph)
            
        return fig, result_message
    
    except Exception as e:
        return go.Figure(), f"Error applying algorithm: {str(e)}"

# Helper function for standard graph visualization
def create_standard_graph_figure(G):
    pos = nx.spring_layout(G)
    
    # Create edges
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    # Create nodes with optional ordering information
    node_colors = []
    node_sizes = []
    node_texts = []
    
    for node in G.nodes():
        if 'order' in G.nodes[node]:
            order = G.nodes[node]['order']
            color = f'hsl({220 - 220 * order / len(G)}, 70%, 50%)'
            node_colors.append(color)
            node_sizes.append(25)
            node_texts.append(f"{node} (order: {order})")
        else:
            node_colors.append('skyblue')
            node_sizes.append(20)
            node_texts.append(node)
    
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        text=node_texts,
        textposition='top center',
        hoverinfo='text',
        marker=dict(showscale=False, color=node_colors, size=node_sizes,
                   line=dict(width=2, color='DarkSlateGrey'))
    )
    
    return go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False, hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            plot_bgcolor='#1E1E1E', paper_bgcolor='#1E1E1E',
            font=dict(color='white'),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

@callback(
    Output('bn-display', 'figure'),
    Output('bn-output', 'children'),
    Input('create-bn', 'n_clicks'),
    State('bn-structure-input', 'value'),
    State('bn-cpt-input', 'value'),
    prevent_initial_call=True
)
def create_bayesian_network(n_clicks, structure_text, cpt_text):
    global global_bn_model
    
    if not structure_text:
        return go.Figure(), "Please enter Bayesian network structure."
    
    try:
        # Parse structure to create directed graph and pgmpy model
        G = nx.DiGraph()
        edges = []
        lines = structure_text.strip().split('\n')
        
        for line in lines:
            # Match pattern "child: [parent1,parent2,...]"
            match = re.match(r'(\w+):\s*\[(.*?)\]', line)
            if match:
                child = match.group(1)
                parents = [p.strip() for p in match.group(2).split(',') if p.strip()]
                
                # Add node if it doesn't exist
                if child not in G.nodes():
                    G.add_node(child)
                
                # Add edges from parents to child
                for parent in parents:
                    if parent not in G.nodes():
                        G.add_node(parent)
                    G.add_edge(parent, child)
                    edges.append((parent, child))
        
        if not G.nodes():
            return go.Figure(), "No valid Bayesian network structure found. Please check your input format."
        
        # Create pgmpy BayesianNetwork model
        model = BayesianNetwork(edges)
        
        # CPT parsing section
        try:
            # Parse CPT information using yaml
            yaml_dict = yaml.safe_load(cpt_text)
            
            # First, collect all nodes and their possible states
            node_states = {}
            for node, cpt_info in yaml_dict.items():
                # Get node name
                if '|' in node:
                    node_name, _ = node.split('|')
                    node_name = node_name.strip()
                else:
                    node_name = node.strip()
                
                # For nodes without parents, we can directly extract states
                if '|' not in node:
                    node_states[node_name] = list(cpt_info.keys())
            
            # Ensure all nodes have state names defined (default to ['true', 'false'])
            for node in G.nodes():
                if node not in node_states:
                    node_states[node] = ['true', 'false']
            
            # Process CPTs for each node
            cpds = []
            for node, cpt_info in yaml_dict.items():
                # Get node name and parents
                if '|' in node:
                    node_name, parents_str = node.split('|')
                    parents = parents_str.split(',')
                else:
                    node_name = node
                    parents = []
                
                node_name = node_name.strip()
                parents = [p.strip() for p in parents]
                
                # Create TabularCPD
                if not parents:  # No parents, like "rain"
                    states = node_states[node_name]
                    values = [[cpt_info[state]] for state in states]  # Make each value a column
                    cpd = TabularCPD(
                        variable=node_name,
                        variable_card=len(states),
                        values=values,  # This gives shape (n, 1) as required
                        state_names={node_name: states}
                    )
                    cpds.append(cpd)
                else:
                    # For nodes with parents (conditional probabilities)
                    var_states = node_states[node_name] if node_name in node_states else ['true', 'false']
                    
                    # Create state_names dictionary with consistent states
                    state_names = {node_name: var_states}
                    for parent in parents:
                        parent_states = node_states[parent] if parent in node_states else ['true', 'false']
                        state_names[parent] = parent_states
                    
                    # Extract probabilities from the CPT format
                    evidence_card = [len(state_names[parent]) for parent in parents]
                    
                    # Create a properly shaped matrix for the CPD values
                    values = np.zeros((len(var_states), np.prod(evidence_card, dtype=int)))
                    
                    # Fill in default values (uniform distribution)
                    for i in range(len(var_states)):
                        for j in range(int(np.prod(evidence_card))):
                            values[i][j] = 1.0 / len(var_states)
                    
                    # Extract values from the CPT info
                    for cond, prob in cpt_info.items():
                        if '|' in cond:
                            var_state, parent_states = cond.split('|')
                            var_state = var_state.strip()
                            parent_states = [p.strip() for p in parent_states.split(',')]
                            
                            # Get row index from variable state
                            if var_state in var_states:
                                row_idx = var_states.index(var_state)
                            else:
                                continue  # Skip if state doesn't match
                            
                            # Calculate column index based on parent states
                            col_idx = 0
                            multiplier = 1
                            for i, parent in enumerate(reversed(parents)):
                                parent_state = parent_states[len(parents)-i-1]
                                if parent in state_names and parent_state in state_names[parent]:
                                    parent_state_idx = state_names[parent].index(parent_state)
                                    col_idx += parent_state_idx * multiplier
                                else:
                                    # Skip if parent state doesn't match
                                    continue
                                multiplier *= len(state_names[parent])
                            
                            # Set the probability value
                            values[row_idx][col_idx] = prob
                    
                    # Normalize each column if needed
                    for col in range(values.shape[1]):
                        col_sum = np.sum(values[:, col])
                        if col_sum > 0 and abs(col_sum - 1.0) > 1e-6:
                            values[:, col] = values[:, col] / col_sum
                    
                    # Create TabularCPD
                    cpd = TabularCPD(
                        variable=node_name,
                        variable_card=len(var_states),
                        values=values,
                        evidence=parents,
                        evidence_card=evidence_card,
                        state_names=state_names
                    )
                    cpds.append(cpd)
            
            # Add CPDs to model
            for cpd in cpds:
                model.add_cpds(cpd)
            
            global_bn_model = model
            bn_info = "Bayesian Network model created successfully with pgmpy!"
            
        except Exception as e:
            bn_info = f"Error parsing CPT information: {str(e)}"
            bn_info += "\nUsing simplified probabilities instead."
            
            # Create default CPDs for all nodes (fallback)
            for node in G.nodes():
                parents = list(G.predecessors(node))
                evidence_card = [2] * len(parents) if parents else []
                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=[[0.5], [0.5]] if not parents else 
                        [[0.5 for _ in range(2**len(parents))], 
                        [0.5 for _ in range(2**len(parents))]],
                    evidence=parents,
                    evidence_card=evidence_card,
                    state_names={
                        node: ['true', 'false'],
                        **{parent: ['true', 'false'] for parent in parents}
                    }
                )
                model.add_cpds(cpd)
            
            global_bn_model = model
        
        # Create positions for visualization (layered for directed graph)
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            # Fall back to spring layout if graphviz is not available
            pos = nx.spring_layout(G)
        
        # Create edges with arrows
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create separate arrow trace
        arrow_trace = go.Scatter(
            x=[],
            y=[],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=10,
                color='white',
                angleref='previous',
                standoff=15
            ),
            hoverinfo='none'
        )
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Calculate points for the edge
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
            
            # Calculate position for the arrow (80% of the way from start to end)
            arrow_x = x0 * 0.2 + x1 * 0.8
            arrow_y = y0 * 0.2 + y1 * 0.8
            
            # Calculate angle for the arrow
            angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
            
            arrow_trace['x'] += (arrow_x,)
            arrow_trace['y'] += (arrow_y,)
            
        # Create nodes
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            text=list(G.nodes()),
            textposition='top center',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color='lightgreen',
                size=25,
                line=dict(width=2, color='DarkSlateGrey')
            )
        )
            
        # Create the figure
        fig = go.Figure(data=[edge_trace, arrow_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            plot_bgcolor='#1E1E1E',
                            paper_bgcolor='#1E1E1E',
                            font=dict(color='white'),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
        
        model_info = ""
        try:
            if global_bn_model:
                model_info = "\n\nModel Information:"
                model_info += "\nVariables: " + ", ".join(global_bn_model.nodes())
                model_info += "\nEdges: " + ", ".join([f"{e[0]}->{e[1]}" for e in global_bn_model.edges()])
                model_info += "\n\nThe model is ready for inference!"
        except Exception as e:
            model_info = f"\nError with model: {str(e)}"
        
        return fig, f"Bayesian network created with {len(G.nodes())} nodes and {len(G.edges())} edges. {bn_info}{model_info}"
    
    except Exception as e:
        return go.Figure(), f"Error creating Bayesian network: {str(e)}"

# Updated callback for running inference using pgmpy
@callback(
    Output('bn-output', 'children', allow_duplicate=True),
    Input('run-inference', 'n_clicks'),
    State('bn-query', 'value'),
    State('bn-evidence', 'value'),
    State('bn-output', 'children'),
    prevent_initial_call=True
)
def run_inference(n_clicks, query, evidence, current_output):
    global global_bn_model
    
    if not query:
        return current_output + "\n\nPlease specify a query."
    
    if global_bn_model is None:
        return current_output + "\n\nPlease create a Bayesian network first."
    
    try:
        # Parse query
        query_parts = query.strip().split('=')
        if len(query_parts) != 2:
            return current_output + "\n\nInvalid query format. Use format: variable=value"
        
        query_var = query_parts[0].strip()
        query_val = query_parts[1].strip().lower()
        
        # Parse evidence
        evidence_dict = {}
        if evidence:
            evidence_items = evidence.split(',')
            for item in evidence_items:
                if '=' in item:
                    var, val = item.split('=')
                    evidence_dict[var.strip()] = val.strip().lower()
        
        # Run inference
        inference = VariableElimination(global_bn_model)
        result = inference.query(variables=[query_var], evidence=evidence_dict)
        
        # Format the result
        response = f"Query: P({query_var}={query_val}"
        if evidence_dict:
            response += f" | {', '.join([f'{k}={v}' for k,v in evidence_dict.items()])}"
        response += ")\n\n"
        
        # Get the probability distribution from the result
        result_str = str(result[query_var])
        response += f"Probability Distribution:\n{result_str}\n"
        
        # Try to extract the specific probability for the queried value
        try:
            if query_val in result[query_var].state_names[query_var]:
                idx = result[query_var].state_names[query_var].index(query_val)
                prob = result[query_var].values[idx]
                response += f"\nP({query_var}={query_val}) = {prob:.4f}"
        except Exception as e:
            response += f"\nCouldn't extract specific probability: {str(e)}"
        
        return current_output + "\n\n" + "-"*40 + "\n" + response
    
    except Exception as e:
        return current_output.split('\n\n')[0] + f"\n\nError running inference:\n{str(e)}\n\n"