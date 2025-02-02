from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import requests
import plotly.express as px
import pandas as pd

# Initialize the Dash app with a modern theme
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Layout of the app
app.layout = dbc.Container(
    [
        # Navbar
        dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src="https://i0.wp.com/static.loxgamestudio.com/securitybot/LoxSecurityBot.png", height="30px")),
                                dbc.Col(dbc.NavbarBrand("Bot Detection Dashboard", className="ms-2")),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="#",
                        style={"textDecoration": "none"},
                    ),
                    dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                ]
            ),
            color="dark",
            dark=True,
            className="mb-4",
        ),

        # Main Content
        dbc.Row(
            [
                dbc.Col(
                    [
                        # Input Form Card
                        dbc.Card(
                            [
                                dbc.CardHeader("Enter Tweet Details", className="text-white bg-primary"),
                                dbc.CardBody(
                                    [
                                        dbc.Form(
                                            [
                                                dbc.Label("Tweet Text:", className="mb-2"),
                                                dcc.Textarea(
                                                    id="tweet-text",
                                                    placeholder="Enter the tweet text...",
                                                    className="mb-3",
                                                    style={"height": "100px"},
                                                ),
                                                dbc.Label("Hashtags (comma-separated):", className="mb-2"),
                                                dcc.Input(
                                                    id="hashtags",
                                                    placeholder="e.g., #AI, #MachineLearning",
                                                    className="mb-3",
                                                ),
                                                dbc.Label("Retweet Count:", className="mb-2"),
                                                dcc.Input(
                                                    id="retweet-count",
                                                    type="number",
                                                    placeholder="Enter retweet count",
                                                    className="mb-3",
                                                ),
                                                dbc.Label("Mention Count:", className="mb-2"),
                                                dcc.Input(
                                                    id="mention-count",
                                                    type="number",
                                                    placeholder="Enter mention count",
                                                    className="mb-3",
                                                ),
                                                dbc.Label("Follower Count:", className="mb-2"),
                                                dcc.Input(
                                                    id="follower-count",
                                                    type="number",
                                                    placeholder="Enter follower count",
                                                    className="mb-3",
                                                ),
                                                dbc.Label("Verified User:", className="mb-2"),
                                                dcc.Dropdown(
                                                    id="verified",
                                                    options=[
                                                        {"label": "Yes", "value": 1},
                                                        {"label": "No", "value": 0},
                                                    ],
                                                    placeholder="Select if the user is verified",
                                                    className="mb-3",
                                                ),
                                                dbc.Button(
                                                    "Predict",
                                                    id="predict-button",
                                                    color="primary",
                                                    className="w-100 mb-3",
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            className="shadow-lg mb-4",
                        ),
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        # Prediction Result Card
                        dbc.Card(
                            [
                                dbc.CardHeader("Prediction Results", className="text-white bg-success"),
                                dbc.CardBody(
                                    [
                                        html.Div(id="prediction-output", className="text-center"),
                                        dcc.Graph(id="probability-graph", style={"height": "300px"}),
                                    ]
                                ),
                            ],
                            className="shadow-lg",
                        ),
                    ],
                    md=6,
                ),
            ],
            className="g-4",
        ),
    ],
    fluid=True,
    style={"padding": "20px"},
)

# Callback to handle prediction
@app.callback(
    [Output("prediction-output", "children"), Output("probability-graph", "figure")],
    Input("predict-button", "n_clicks"),
    State("tweet-text", "value"),
    State("hashtags", "value"),
    State("retweet-count", "value"),
    State("mention-count", "value"),
    State("follower-count", "value"),
    State("verified", "value"),
    prevent_initial_call=True,  # Prevents the callback from running on page load
)
def predict_bot(n_clicks, tweet_text, hashtags, retweet_count, mention_count, follower_count, verified):
    if not tweet_text:
        return "Please enter tweet details and click Predict.", {}

    try:
        # Prepare input data for API
        input_data = {
            "Tweet": tweet_text,
            "Retweet Count": retweet_count or 0,
            "Follower Count": follower_count or 0,
            "Verified": verified or 0,
            "Hashtags": hashtags or "",  # Optional field
            "Mention Count": mention_count or 0,  # Optional field
        }

        # Send request to Flask API
        response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
        result = response.json()

        if "error" in result:
            return f"Error: {result['error']}", {}

        # Display prediction and probabilities
        prediction = "Bot" if result["prediction"] == 1 else "Not a Bot"
        probabilities = result["probabilities"][0]

        # Create a bar chart for probabilities
        df = pd.DataFrame({"Category": ["Bot", "Not a Bot"], "Probability": probabilities})
        fig = px.bar(df, x="Category", y="Probability", text="Probability", color="Category", height=300)
        fig.update_traces(texttemplate="%{y:.2f}", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)

        return [
            html.H4(f"Prediction: {prediction}", className="mb-2"),
            html.P(f"Confidence: {max(probabilities):.2f}", className="mb-0"),
        ], fig
    except Exception as e:
        return f"Error: {str(e)}", {}

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
