from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import requests

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Bot Detection Dashboard", className="text-center mb-4"),
            html.P("Enter tweet details to detect if it's from a bot.", className="text-center mb-4"),
            dbc.Form([
                dbc.Label("Tweet Text:", className="mb-2"),
                dcc.Textarea(id='tweet-text', placeholder='Enter the tweet text', className="mb-3", style={'height': '100px'}),

                dbc.Label("Hashtags (comma-separated):", className="mb-2"),
                dcc.Input(id='hashtags', placeholder='e.g., #AI, #MachineLearning', className="mb-3"),

                dbc.Label("Retweet Count:", className="mb-2"),
                dcc.Input(id='retweet-count', type='number', placeholder='Enter the retweet count', className="mb-3"),

                dbc.Label("Mention Count:", className="mb-2"),
                dcc.Input(id='mention-count', type='number', placeholder='Enter the mention count', className="mb-3"),

                dbc.Label("Follower Count:", className="mb-2"),
                dcc.Input(id='follower-count', type='number', placeholder='Enter the follower count', className="mb-3"),

                dbc.Label("Verified User:", className="mb-2"),
                dcc.Dropdown(
                    id='verified',
                    options=[
                        {'label': 'Yes', 'value': 1},
                        {'label': 'No', 'value': 0}
                    ],
                    placeholder='Select if the user is verified',
                    className="mb-3"
                ),

                dbc.Button('Predict', id='predict-button', color='primary', className="mb-3"),
            ]),
            html.Div(id='prediction-output', className="mt-4")
        ], width=8)
    ], justify="center")
])

# Callback to handle prediction
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('tweet-text', 'value'),
    State('hashtags', 'value'),
    State('retweet-count', 'value'),
    State('mention-count', 'value'),
    State('follower-count', 'value'),
    State('verified', 'value'),
    prevent_initial_call=True  # Prevents the callback from running on page load
)
def predict_bot(n_clicks, tweet_text, hashtags, retweet_count, mention_count, follower_count, verified):
    if not tweet_text:
        return "Please enter tweet details and click Predict."

    try:
        # Prepare input data for API
        input_data = {
            'Tweet': tweet_text,
            'Retweet Count': retweet_count or 0,
            'Follower Count': follower_count or 0,
            'Verified': verified or 0,
            'Hashtags': hashtags or "",  # Optional field
            'Mention Count': mention_count or 0,  # Optional field
        }

        # Send request to Flask API
        response = requests.post('http://127.0.0.1:5000/predict', json=input_data)
        result = response.json()

        if 'error' in result:
            return f"Error: {result['error']}"

        # Display prediction and probabilities
        prediction = "Bot" if result['prediction'] == 1 else "Not a Bot"
        probabilities = result['probabilities'][0]
        return [
            html.H4(f"Prediction: {prediction}", className="mb-2"),
            html.P(f"Probabilities: {probabilities}", className="mb-0")
        ]
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)