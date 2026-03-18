# ════════════════════════════════════════════════════════════════════════
# THE VOICEBOX
# A Databricks App for creative LLM fun

import dash
from dash import html, dcc, callback, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import psycopg
import os
import random
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from databricks.sdk import WorkspaceClient
from psycopg import sql
from psycopg_pool import ConnectionPool

# ── Model Configuration ─────────────────────────────────────────────────

MODELS = {
    "databricks-meta-llama-3-3-70b-instruct": {
        "name": "Llama 3.3 70B",
        "icon": "\U0001f999",
        "color": "#667eea",
        "tagline": "Meta? More like Mediocre...",
    },
    "databricks-claude-sonnet-4": {
        "name": "Claude Sonnet 4",
        "icon": "\U0001f3ad",
        "color": "#e07b39",
        "tagline": "Anthropic's attmempt at an LLM...",
    }
}

MODEL_IDS = list(MODELS.keys())

# ── Task Definitions ────────────────────────────────────────────────────

TASKS = {
    "roast": {
        "label": "\U0001f525 Roast Me",
        "short": "Roast Battle",
        "desc": "Roast Databricks Users - LLM Style",
    },
    "joke": {
        "label": "\U0001f923 Joke Telling",
        "short": "Joke-Off",
        "desc": "LLM tells a joke.",
    },
    "poem": {
        "label": "\U0001f4dd Poem Writing",
        "short": "Poetry Slam",
        "desc": "Craft a creative, memorable poem",
    },
}



# ── LLM Interaction ─────────────────────────────────────────────────────

_ws_client = None

def get_ws_client():
    global _ws_client
    if _ws_client is None:
        _ws_client = WorkspaceClient()
    return _ws_client

# Database connection setup

postgres_password = None
last_password_refresh = 0
connection_pool = None

def refresh_oauth_token():
    """Refresh OAuth token if expired."""
    global postgres_password, last_password_refresh
    if postgres_password is None or time.time() - last_password_refresh > 900:
        print("Refreshing PostgreSQL OAuth token")
        try:
            postgres_password = get_ws_client().config.oauth_token().access_token
            last_password_refresh = time.time()
        except Exception as e:
            print(f"❌ Failed to refresh OAuth token: {str(e)}")
            return False
    return True
    
def get_connection_pool():
    """Get or create the connection pool."""
    global connection_pool
    if connection_pool is None:
        refresh_oauth_token()
        conn_string = (
            f"dbname={os.getenv('PGDATABASE')} "
            f"user={os.getenv('PGUSER')} "
            f"password={postgres_password} "
            f"host={os.getenv('PGHOST')} "
            f"port={os.getenv('PGPORT')} "
            f"sslmode={os.getenv('PGSSLMODE', 'require')} "
            f"application_name={os.getenv('PGAPPNAME')}"
        )
        connection_pool = ConnectionPool(conn_string, min_size=2, max_size=10)
    return connection_pool

def get_connection():
    """Get a connection from the pool."""
    global connection_pool
    
    # Recreate pool if token expired
    if postgres_password is None or time.time() - last_password_refresh > 900:
        if connection_pool:
            connection_pool.close()
            connection_pool = None
    
    return get_connection_pool().connection()

def get_schema_name():
    """Get the schema name in the format {PGAPPNAME}_schema_{PGUSER}."""
    pgappname = os.getenv("PGAPPNAME", "my_app")
    pguser = os.getenv("PGUSER", "").replace('-', '')
    return f"voicebox"

def init_database():
    """Initialize database schema and table."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                schema_name = get_schema_name()
                
                cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name)))
                print(f"✅ Schema '{schema_name}' created or already exists.")
                cur.execute(sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {}.llm_outputs (
                        timestamp DOUBLE PRECISION,
                        model_id TEXT,
                        system_prompt TEXT,
                        user_prompt TEXT,
                        model_output TEXT,
                        input_tokens INTEGER,
                        output_tokens INTEGER
                    )
                """).format(sql.Identifier(schema_name)))
                print(f"✅ Table '{schema_name}.llm_outputs' created or already exists.")
                conn.commit()
                return True
    except Exception as e:
        print(f"Database initialization error: {e}")
        return False
    
def add_item_to_db(model_id, system_prompt, user_prompt, model_output, input_tokens, output_tokens):
    """Add a new item to the database."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                schema = get_schema_name()
                cur.execute(sql.SQL("""
                    INSERT INTO {}.llm_outputs (
                        timestamp, model_id, system_prompt, user_prompt, model_output, input_tokens, output_tokens
                    ) VALUES (
                        EXTRACT(EPOCH FROM NOW()), %s, %s, %s, %s, %s, %s
                    )
                """).format(sql.Identifier(schema)), (model_id, system_prompt, user_prompt, model_output, input_tokens, output_tokens))
                conn.commit()
                return True
    except Exception as e:
        print(f"Add item error: {e}")
        return False
    
def build_prompt(task_id, model_name, opponent_name=None):
    prompts = {
        "roast": (
            f"Write a hilarious, creative roast of a databricks professional. Talk about them using the platform in a sub-optimal capacity. "
            f"Keep it to 2-3 sentences. Be as mean as possible, but dont curse or be racist! Make it fun to read aloud!"
        ),
        "joke": (
            f"Tell the funniest, most original joke you can think of. "
            f"It can be a tech joke, a pun, observational humor, or absurdist comedy. Aim for genuine laughs. Make it fun to read aloud!"
        ),
        "poem": (
            f"Write a short, creative poem (3-6 lines) about technology, AI, data, or the future. "
            f"Make it memorable and show your unique voice. Make it fun to read aloud! "
        )
    }
    return prompts.get(task_id, "Tell me something impressive about yourself.")

def call_model(endpoint_name, system_prompt, user_prompt, max_tokens=500):
    """Call a Databricks Foundation Model using REST API (avoids SDK serialization issues)."""
    try:
        w = get_ws_client()
        url = f"{w.config.host}/serving-endpoints/{endpoint_name}/invocations"
        headers = {"Content-Type": "application/json"}
        headers.update(w.config.authenticate())
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.9,
        }
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # Extract input/output token counts if available
        input_tokens = data.get("usage", {}).get("prompt_tokens", None)
        output_tokens = data.get("usage", {}).get("completion_tokens", None)
        model_output = data["choices"][0]["message"]["content"]
        add_item_to_db(
            endpoint_name,
            system_prompt,
            user_prompt,
            model_output,
            input_tokens,
            output_tokens
        )
        return model_output
    except Exception as e:
        return f"*[Response generation failed: {e}]*"

SYSTEM_PROMPT = (
    "You are an AI creativity playground on Databricks. I want you to be the sassiest AI ever. Its okay to offend."
    "Give your absolute best, most creative response. Be concise but impactful. Show personality! Make answers fun to read aloud!"
)

# ── Dash App Layout & Callbacks ─────────────────────────────────────────

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.VAPOR])

# Initialize database
if not init_database():
    print("Failed to initialize database")
    
app.layout = dbc.Container([
    html.H1("📢 VOICEBOX", style={"textAlign": "center", "marginTop": "2rem"}),
    html.P("Pick a task and a model. Leave your feelings at the door.", style={"textAlign": "center"}),
    dbc.Row([
        dbc.Col([
            html.Label("Select Task"),
            dcc.Dropdown(
                id="task-dropdown",
                options=[{"label": v["label"], "value": k} for k, v in TASKS.items()],
                value="roast",
                clearable=False,
            ),
        ], width=6),
        dbc.Col([
            html.Label("Select Model"),
            dcc.Dropdown(
                id="model-dropdown",
                options=[{"label": MODELS[m]["name"], "value": m} for m in MODEL_IDS],
                value=MODEL_IDS[0],
                clearable=False,
            ),
        ], width=6),
    ], style={"marginTop": "2rem"}),
    dbc.Button("Generate", id="generate-btn", color="warning", className="mt-3"),
    html.Div(id="result-div", style={"marginTop": "2rem"}),
])

@callback(
    Output("result-div", "children"),
    Input("generate-btn", "n_clicks"),
    State("task-dropdown", "value"),
    State("model-dropdown", "value"),
    prevent_initial_call=False,
)
def run_funbox(n_clicks, task_id, model_id):
    if not model_id or not task_id:
        return dbc.Alert("Please select a task and a model.", color="danger")
    model = MODELS[model_id]
    prompt = build_prompt(task_id, model["name"])
    response = call_model(model_id, SYSTEM_PROMPT, prompt)
    return dbc.Card([
        dbc.CardHeader(f"{model['icon']} {model['name']}"),
        dbc.CardBody([
            html.P(
                response,
                style={
                    "background": model["color"],
                    "color": "white",
                    "padding": "1rem",
                    "borderRadius": "8px"
                }
            ),
            html.Small(model["tagline"]),
        ])
    ], style={"marginBottom": "2rem"})

# ── Databricks App Entrypoint ───────────────────────────────────────────

def main():
    app.run(debug=True)

if __name__ == "__main__":
    main()