<<<<<<< HEAD
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import os
import datetime

# Create a directory to store feedback if it doesn't exist
feedback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_data")
os.makedirs(feedback_dir, exist_ok=True)

# Function to read code from files
def get_file_content(filename):
    try:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        return f"Error: File {filename} not found"
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Read code from actual files
code_samples = {
    'evo_algo': get_file_content('evo_algo.py'),
    'gen_algo': get_file_content('gen_algo.py'),
    'prob_rea': get_file_content('prob_rea.py')
}

# Function to save feedback to a text file
def save_feedback_to_file(name, feedback, code_type):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"feedback_{code_type}_{timestamp}.txt"
    file_path = os.path.join(feedback_dir, filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reviewer: {name}\n")
        f.write(f"Code Reviewed: {code_type}\n")
        f.write("-" * 50 + "\n")
        f.write("Feedback:\n")
        f.write(feedback)
    
    return file_path

def get_all_feedback():
    """Read all feedback files and return as a list of dictionaries"""
    feedbacks = []
    if os.path.exists(feedback_dir):
        for filename in sorted(os.listdir(feedback_dir), reverse=True):
            if filename.startswith('feedback_') and filename.endswith('.txt'):
                file_path = os.path.join(feedback_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Parse the content to extract metadata
                    lines = content.split('\n')
                    timestamp = lines[0].replace("Timestamp: ", "")
                    reviewer = lines[1].replace("Reviewer: ", "")
                    code_type = lines[2].replace("Code Reviewed: ", "")
                    
                    # Get feedback content (everything after the separator)
                    separator_index = content.find('-' * 50) + len('-' * 50) + 1
                    feedback_text = content[separator_index:].strip()
                    
                    feedbacks.append({
                        'timestamp': timestamp,
                        'reviewer': reviewer,
                        'code_type': code_type,
                        'feedback': feedback_text,
                        'filename': filename
                    })
                except Exception as e:
                    print(f"Error reading feedback file {filename}: {str(e)}")
    
    return feedbacks

layout = html.Div([
    html.H2("Code Review and Feedback", style={'textAlign': 'center'}),
    html.P("Welcome to the code review and feedback section. Please select a code sample to review and provide your feedback. For detailed feedback and contributions to code feel free to reach out on email. You can also check out the complete project on GitHub."),

    html.P("Email: m.jamil.29409@khi.iba.edu.pk"),
    html.P("GitHub: https://github.com/MusabbinJamil/pythonanywhere.git"),

    html.Hr(),
    
    # Code display section
    html.Div([
        html.H3("Code Sample for Review"),
        html.Label("Select code to review:"),
        dcc.Dropdown(
            id='code-selector',
            options=[
                {'label': 'Evolutionary Algorithm', 'value': 'evo_algo'},
                {'label': 'Genetic Algorithm', 'value': 'gen_algo'},
                {'label': 'Probabilistic Reasoning', 'value': 'prob_rea'}
            ],
            value='evo_algo',
            style={'color': 'black', 'marginBottom': '15px'}
        ),
        
        # Uneditable code display - using html.Pre instead of SyntaxHighlighter
        html.Div([
            html.Label("Code:"),
            html.Pre(
                id='code-display',
                style={
                    'backgroundColor': '#2E2E2E',
                    'color': 'white',
                    'padding': '15px',
                    'borderRadius': '5px',
                    'marginBottom': '20px',
                    'fontFamily': 'monospace',
                    'whiteSpace': 'pre-wrap',
                    'maxHeight': '400px',
                    'overflowY': 'auto'
                }
            )
        ]),
    ]),
    
    html.Hr(),
    
    # Feedback form
    html.Div([
        html.H3("Submit Your Feedback"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Your Name:"),
                dcc.Input(
                    id='feedback-name',
                    type='text',
                    placeholder='Enter your name',
                    style={
                        'width': '100%',
                        'padding': '8px',
                        'color': 'black',
                        'marginBottom': '15px'
                    }
                ),
            ], width=6)
        ]),
        
        html.Label("Your Feedback:"),
        dcc.Textarea(
            id='feedback-text',
            placeholder='Please provide your feedback on the code...',
            style={
                'width': '100%',
                'height': '150px',
                'padding': '10px',
                'color': 'black',
                'marginBottom': '15px'
            }
        ),
        
        html.Button('Submit Feedback', 
                   id='submit-feedback', 
                   style={
                       'backgroundColor': '#4CAF50',
                       'color': 'white',
                       'padding': '10px 15px',
                       'border': 'none',
                       'borderRadius': '4px',
                       'cursor': 'pointer'
                   }),
        
        html.Div(id='feedback-submission-result'),

        html.Hr(),
    
        # Feedback display section
        html.Div([
            html.H3("Recent Feedback Submissions"),
            html.Button('Refresh Feedback', 
                    id='refresh-feedback', 
                    style={
                        'backgroundColor': '#4682B4',
                        'color': 'white',
                        'padding': '8px 12px',
                        'border': 'none',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'marginBottom': '15px'
                    }),
            html.Div(id='feedback-display')
        ]),
    ]),
], style={'padding': '20px'})

@callback(
    Output('code-display', 'children'),
    Input('code-selector', 'value')
)
def update_code_display(selected_code):
    return code_samples.get(selected_code, "No code selected")

@callback(
    Output('feedback-submission-result', 'children'),
    Input('submit-feedback', 'n_clicks'),
    State('feedback-name', 'value'),
    State('feedback-text', 'value'),
    State('code-selector', 'value'),
    prevent_initial_call=True
)
def submit_feedback(n_clicks, name, feedback, code_type):
    if not name or not feedback:
        return html.Div("Please fill in all fields", style={'color': 'red'})
    
    if not code_type:
        return html.Div("Please select a code sample to review", style={'color': 'red'})
    
    try:
        # Save feedback to a file
        saved_path = save_feedback_to_file(name, feedback, code_type)
        
        # Display success message
        return html.Div([
            html.P(f"Thank you {name} for your feedback!", style={'color': 'green'}),
            html.P(f"Your feedback has been saved successfully.", style={'color': 'green'}),
            html.P(f"File saved: {os.path.basename(saved_path)}", style={'fontSize': '0.9em', 'color': '#88c999'})
        ], style={'marginTop': '10px'})
    except Exception as e:
        return html.Div(f"Error saving feedback: {str(e)}", style={'color': 'red'})

@callback(
    Output('feedback-display', 'children'),
    [Input('refresh-feedback', 'n_clicks'),
     Input('feedback-submission-result', 'children')],
)
def update_feedback_display(n_clicks, submission_result):
    feedbacks = get_all_feedback()
    
    if not feedbacks:
        return html.Div("No feedback submissions yet.", style={'fontStyle': 'italic'})
    
    feedback_cards = []
    # Show the 10 most recent feedbacks
    for feedback in feedbacks[:10]:  
        card = dbc.Card(
            dbc.CardBody([
                html.H5(f"Feedback on {feedback['code_type']}", className="card-title"),
                html.H6(f"By: {feedback['reviewer']}", className="card-subtitle"),
                html.P(f"Submitted: {feedback['timestamp']}", className="text-muted"),
                html.Hr(style={'margin': '10px 0'}),
                html.P(feedback['feedback'], style={'whiteSpace': 'pre-wrap'}),
            ]),
            className="mb-3",
        )
        feedback_cards.append(card)
    
=======
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import os
import datetime

# Create a directory to store feedback if it doesn't exist
feedback_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_data")
os.makedirs(feedback_dir, exist_ok=True)

# Function to read code from files
def get_file_content(filename):
    try:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        return f"Error: File {filename} not found"
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Read code from actual files
code_samples = {
    'evo_algo': get_file_content('evo_algo.py'),
    'gen_algo': get_file_content('gen_algo.py'),
    'prob_rea': get_file_content('prob_rea.py')
}

# Function to save feedback to a text file
def save_feedback_to_file(name, feedback, code_type):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"feedback_{code_type}_{timestamp}.txt"
    file_path = os.path.join(feedback_dir, filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reviewer: {name}\n")
        f.write(f"Code Reviewed: {code_type}\n")
        f.write("-" * 50 + "\n")
        f.write("Feedback:\n")
        f.write(feedback)
    
    return file_path

def get_all_feedback():
    """Read all feedback files and return as a list of dictionaries"""
    feedbacks = []
    if os.path.exists(feedback_dir):
        for filename in sorted(os.listdir(feedback_dir), reverse=True):
            if filename.startswith('feedback_') and filename.endswith('.txt'):
                file_path = os.path.join(feedback_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Parse the content to extract metadata
                    lines = content.split('\n')
                    timestamp = lines[0].replace("Timestamp: ", "")
                    reviewer = lines[1].replace("Reviewer: ", "")
                    code_type = lines[2].replace("Code Reviewed: ", "")
                    
                    # Get feedback content (everything after the separator)
                    separator_index = content.find('-' * 50) + len('-' * 50) + 1
                    feedback_text = content[separator_index:].strip()
                    
                    feedbacks.append({
                        'timestamp': timestamp,
                        'reviewer': reviewer,
                        'code_type': code_type,
                        'feedback': feedback_text,
                        'filename': filename
                    })
                except Exception as e:
                    print(f"Error reading feedback file {filename}: {str(e)}")
    
    return feedbacks

layout = html.Div([
    html.H2("Code Review and Feedback", style={'textAlign': 'center'}),
    html.P("Welcome to the code review and feedback section. Please select a code sample to review and provide your feedback. For detailed feedback and contributions to code feel free to reach out on email. You can also check out the complete project on GitHub."),

    html.P("Email: m.jamil.29409@khi.iba.edu.pk"),
    html.P("GitHub: https://github.com/MusabbinJamil/pythonanywhere.git"),

    html.Hr(),
    
    # Code display section
    html.Div([
        html.H3("Code Sample for Review"),
        html.Label("Select code to review:"),
        dcc.Dropdown(
            id='code-selector',
            options=[
                {'label': 'Evolutionary Algorithm', 'value': 'evo_algo'},
                {'label': 'Genetic Algorithm', 'value': 'gen_algo'},
                {'label': 'Probabilistic Reasoning', 'value': 'prob_rea'}
            ],
            value='evo_algo',
            style={'color': 'black', 'marginBottom': '15px'}
        ),
        
        # Uneditable code display - using html.Pre instead of SyntaxHighlighter
        html.Div([
            html.Label("Code:"),
            html.Pre(
                id='code-display',
                style={
                    'backgroundColor': '#2E2E2E',
                    'color': 'white',
                    'padding': '15px',
                    'borderRadius': '5px',
                    'marginBottom': '20px',
                    'fontFamily': 'monospace',
                    'whiteSpace': 'pre-wrap',
                    'maxHeight': '400px',
                    'overflowY': 'auto'
                }
            )
        ]),
    ]),
    
    html.Hr(),
    
    # Feedback form
    html.Div([
        html.H3("Submit Your Feedback"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Your Name:"),
                dcc.Input(
                    id='feedback-name',
                    type='text',
                    placeholder='Enter your name',
                    style={
                        'width': '100%',
                        'padding': '8px',
                        'color': 'black',
                        'marginBottom': '15px'
                    }
                ),
            ], width=6)
        ]),
        
        html.Label("Your Feedback:"),
        dcc.Textarea(
            id='feedback-text',
            placeholder='Please provide your feedback on the code...',
            style={
                'width': '100%',
                'height': '150px',
                'padding': '10px',
                'color': 'black',
                'marginBottom': '15px'
            }
        ),
        
        html.Button('Submit Feedback', 
                   id='submit-feedback', 
                   style={
                       'backgroundColor': '#4CAF50',
                       'color': 'white',
                       'padding': '10px 15px',
                       'border': 'none',
                       'borderRadius': '4px',
                       'cursor': 'pointer'
                   }),
        
        html.Div(id='feedback-submission-result'),

        html.Hr(),
    
        # Feedback display section
        html.Div([
            html.H3("Recent Feedback Submissions"),
            html.Button('Refresh Feedback', 
                    id='refresh-feedback', 
                    style={
                        'backgroundColor': '#4682B4',
                        'color': 'white',
                        'padding': '8px 12px',
                        'border': 'none',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'marginBottom': '15px'
                    }),
            html.Div(id='feedback-display')
        ]),
    ]),
], style={'padding': '20px'})

@callback(
    Output('code-display', 'children'),
    Input('code-selector', 'value')
)
def update_code_display(selected_code):
    return code_samples.get(selected_code, "No code selected")

@callback(
    Output('feedback-submission-result', 'children'),
    Input('submit-feedback', 'n_clicks'),
    State('feedback-name', 'value'),
    State('feedback-text', 'value'),
    State('code-selector', 'value'),
    prevent_initial_call=True
)
def submit_feedback(n_clicks, name, feedback, code_type):
    if not name or not feedback:
        return html.Div("Please fill in all fields", style={'color': 'red'})
    
    if not code_type:
        return html.Div("Please select a code sample to review", style={'color': 'red'})
    
    try:
        # Save feedback to a file
        saved_path = save_feedback_to_file(name, feedback, code_type)
        
        # Display success message
        return html.Div([
            html.P(f"Thank you {name} for your feedback!", style={'color': 'green'}),
            html.P(f"Your feedback has been saved successfully.", style={'color': 'green'}),
            html.P(f"File saved: {os.path.basename(saved_path)}", style={'fontSize': '0.9em', 'color': '#88c999'})
        ], style={'marginTop': '10px'})
    except Exception as e:
        return html.Div(f"Error saving feedback: {str(e)}", style={'color': 'red'})

@callback(
    Output('feedback-display', 'children'),
    [Input('refresh-feedback', 'n_clicks'),
     Input('feedback-submission-result', 'children')],
)
def update_feedback_display(n_clicks, submission_result):
    feedbacks = get_all_feedback()
    
    if not feedbacks:
        return html.Div("No feedback submissions yet.", style={'fontStyle': 'italic'})
    
    feedback_cards = []
    # Show the 10 most recent feedbacks
    for feedback in feedbacks[:10]:  
        card = dbc.Card(
            dbc.CardBody([
                html.H5(f"Feedback on {feedback['code_type']}", className="card-title"),
                html.H6(f"By: {feedback['reviewer']}", className="card-subtitle"),
                html.P(f"Submitted: {feedback['timestamp']}", className="text-muted"),
                html.Hr(style={'margin': '10px 0'}),
                html.P(feedback['feedback'], style={'whiteSpace': 'pre-wrap'}),
            ]),
            className="mb-3",
        )
        feedback_cards.append(card)
    
>>>>>>> master
    return html.Div(feedback_cards)