####################
#### Setting up ####
####################

# ── Imports ───────────────────────────────────────────────

import time
import os
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache

import boto3
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, Output, Input, html, ctx
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv


# ── Environment & AWS setup ───────────────────────────────

script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

ENV = os.getenv('ENV', 'production')

aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_DEFAULT_REGION')

bucket = 'uk-petitions-dashboard'

if ENV != 'local':
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )


# ── S3 loading functions ──────────────────────────────────

def load_csv(filename):
    s3_object = s3_client.get_object(Bucket=bucket, Key=filename)
    df = pd.read_csv(s3_object['Body'])
    return df


def load_geojson(filename):
    s3_object = s3_client.get_object(Bucket=bucket, Key=filename)
    gdf = gpd.read_file(s3_object['Body'])
    return gdf


# ── Cached data loaders ───────────────────────────────────

@lru_cache(maxsize=1)
def get_constituency_geojson():
    if ENV == 'local':
        local_path = script_dir / 'cached_data' / 'constituencies_july_2024.geojson'
        print(f"Loading GeoJSON from local cache: {local_path}")
        constituency_boundaries = gpd.read_file(local_path)
    else:
        constituency_boundaries = load_geojson(
            'static data/constituencies_july_2024.geojson'
        )
    constituency_boundaries = constituency_boundaries[['PCON24CD', 'geometry']]
    constituency_boundaries['geometry'] = constituency_boundaries['geometry'].simplify(0.005)
    return constituency_boundaries


def get_petitions_data():
    if ENV == 'local':
        for delta in [0, 1]:
            date_str = (datetime.now() - timedelta(days=delta)).strftime('%Y%m%d')
            list_path  = script_dir / 'cached_data' / f'petitions_list_{date_str}.csv'
            count_path = script_dir / 'cached_data' / f'petitions_counts_{date_str}.csv'
            if list_path.exists() and count_path.exists():
                print(f"Loading petitions data from local cache for {date_str}...")
                petitions_list  = pd.read_csv(list_path)
                petitions_count = pd.read_csv(count_path)
                return petitions_list, petitions_count
            print(f"No local cache found for {date_str}, trying previous day...")
        raise FileNotFoundError("No petitions data found in local cache for today or yesterday")
    else:
        for delta in [0, 1]:
            date_str = (datetime.now() - timedelta(days=delta)).strftime('%Y%m%d')
            try:
                petitions_list  = load_csv(f'dynamic_data/petitions_list_{date_str}.csv')
                petitions_count = load_csv(f'dynamic_data/petitions_counts_{date_str}.csv')
                print(f"Loaded data for {date_str}")
                return petitions_list, petitions_count
            except s3_client.exceptions.NoSuchKey:
                print(f"No data found for {date_str}, trying previous day...")
        raise FileNotFoundError("No petitions data found for today or yesterday")


def get_population_data():
    if ENV == 'local':
        local_path = script_dir / 'cached_data' / 'pop_estimate_2021.csv'
        print("Loading population data from local cache...")
        return pd.read_csv(local_path)
    return load_csv('static data/pop_estimate_2021.csv')


@lru_cache(maxsize=128)
def get_petition_data(petition_id):
    start = time.time()
    df = petitions_df[petitions_df['petition_id'] == petition_id]
    print(f"Time to retrieve petition data from cache: {time.time() - start:.4f}s")
    return tuple(df.itertuples(index=False))


# ── Data processing ───────────────────────────────────────

print(f"Environment: {ENV}")

print("Loading GeoJSON...")
constituency_boundaries = get_constituency_geojson()

print("Loading petitions data...")
petitions_list, petitions_count = get_petitions_data()

print("Loading population data...")
pop_df = get_population_data()

print("Done loading data.")

# Data pull has missing rows if value should actually be 0 so making sure they get filled with 0
petition_ids = petitions_list[['petition_id']].drop_duplicates()
pcon24cds = petitions_count[['PCON24CD', 'constituency_name']].drop_duplicates()

skeleton_df = petition_ids.merge(pcon24cds, how='cross')

petitions_count = skeleton_df.merge(petitions_count.drop(columns=['constituency_name']), on = ['petition_id', 'PCON24CD'], how = 'left')
petitions_count['signature_count'] = petitions_count['signature_count'].fillna(0).astype(int)

# output_path = script_dir / "cached_data" / "file_to_check.csv"
# petitions_count.to_csv(output_path, index=False)

# Adding data on population
petitions_count = petitions_count.merge(pop_df[['PCON24CD', 'pop']], on= ['PCON24CD'], how='left')

# Adding column on sig per pop
petitions_count['sig_per_pop'] = (petitions_count['signature_count'] / petitions_count['pop']) * 1000

# Adding rank
petitions_count['sig_per_pop_rank'] = petitions_count.groupby('petition_id')['sig_per_pop'].rank(ascending=False, method='min')
petitions_count['percentile_rank_raw'] = (100 - (petitions_count.groupby('petition_id')['signature_count'].rank(pct=True) * 100)).round(1)
petitions_count['percentile_rank_pop'] = (100 - (petitions_count.groupby('petition_id')['sig_per_pop'].rank(pct=True) * 100)).round(1)

# Working out median count for each petition
median_counts = petitions_count.groupby('petition_id')['signature_count'].median().reset_index()
median_counts.columns = ['petition_id', 'median_signature_count']

# Adding median counts to petitions list
petitions_list = petitions_list.merge(median_counts, on='petition_id', how='left')

# Merging petitions count to petitions list
petitions_df = petitions_list.merge(petitions_count, on='petition_id', how='left')

# If total count is less than 10,000, then remove ranking
petitions_df.loc[petitions_df['total_signature_count'] <= 10000, 'percentile_rank_raw'] = np.nan
petitions_df.loc[petitions_df['total_signature_count'] <= 10000, 'percentile_rank_pop'] = np.nan

petition_quantiles = (
    petitions_df
    .groupby('petition_id')['signature_count']
    .quantile(0.95)
    .to_dict()
)


# ── Utility functions ─────────────────────────────────────

def wrap_text(text, width=50):
    """Insert <br> tags into long strings at word boundaries."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(' '.join(current_line))

    return '<br>'.join(lines)


def _render_bar(value, max_val, bar_color, border_color, marker_pct=None):
    bar_width_pct = (value / max_val) * 100 if max_val else 0
    track_children = [
        html.Div(style={
            'width': f'{bar_width_pct}%',
            'backgroundColor': bar_color,
            'border': f'1.5px solid {border_color}',
            'opacity': 0.8,
            'height': '18px',
            'borderRadius': '2px'
        })
    ]
    if marker_pct is not None:
        track_children.append(html.Div(style={
            'position': 'absolute', 'left': f'{marker_pct}%', 'top': '-2px', 'bottom': '-2px',
            'borderLeft': '2px dotted #333'
        }))

    return html.Div([
        html.Div(track_children, style={'position': 'relative', 'flex': '1 1 auto'}),
        html.Span(f"{value:,}", style={
            'marginLeft': '8px', 'fontSize': '11px', 'color': '#333', 'whiteSpace': 'nowrap'
        })
    ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '2px'})


def render_top5_bars(df, value_col, bar_color='#0d6efd', border_color='#0a58ca', secondary_col=None,
                      marker_col=None):
    """Render a top-5 horizontal bar list, bars scaled to value_col's max across the top 5.
    If secondary_col is given, each row also shows what % of that column's value the
    primary value represents (e.g. constituency votes as a % of the petition's total).
    If marker_col is given, each bar gets its own dotted vertical marker at that row's
    marker_col value (e.g. the petition's median signature count across all constituencies).

    Every row has the same fixed height (title area + bar + secondary-line area, whether
    or not secondary_col is used) so that rows line up across two side-by-side charts
    built from this function, even when one has a secondary line and the other doesn't.
    """
    cols = ['petition_title', value_col, 'petition_url']
    if secondary_col:
        cols.append(secondary_col)
    if marker_col:
        cols.append(marker_col)

    top_5 = df.nlargest(5, value_col)[cols].sort_values(value_col, ascending=False).reset_index(drop=True)

    max_val = top_5[value_col].max()

    rows = []
    for _, row in top_5.iterrows():
        if secondary_col:
            pct = (row[value_col] / row[secondary_col] * 100) if row[secondary_col] else 0
            secondary_text = f"{pct:.2f}% of {row[secondary_col]:,} total votes"
        else:
            secondary_text = ''

        marker_pct = (row[marker_col] / max_val) * 100 if marker_col and max_val else None

        children = [
            html.A(
                row['petition_title'],
                href=row['petition_url'],
                target='_blank',
                style={
                    'fontSize': '12px', 'color': '#333', 'textDecoration': 'none',
                    'display': 'block', 'whiteSpace': 'nowrap', 'overflow': 'hidden',
                    'textOverflow': 'ellipsis', 'width': '100%'
                }
            ),
            _render_bar(row[value_col], max_val, bar_color, border_color, marker_pct),
            html.Div(
                secondary_text,
                style={
                    'marginLeft': '8px', 'fontSize': '10px', 'color': '#777',
                    'marginTop': '1px', 'height': '14px', 'visibility': 'visible' if secondary_col else 'hidden'
                }
            )
        ]

        rows.append(html.Div(children, style={'marginBottom': '6px'}))

    return html.Div(rows, style={'padding': '10px 20px'})


####################
#### App setup  ####
####################

# ── Initialise app ────────────────────────────────────────

app = Dash(__name__, external_stylesheets=[dbc.themes.PULSE], suppress_callback_exceptions=True)
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Native dcc.Tabs header is replaced by the nav in the top banner; hide it
               but keep dcc.Tabs itself so its tab-switching/content mount-unmount logic
               (and the Map tab's plotly graph sizing) still works. */
            .tab-container {
                display: none !important;
            }

            /* Consistent sort arrow alignment across all columns */
            .dash-header .column-header--sort {
                display: flex !important;
                flex-direction: row !important;
                justify-content: space-between !important;
                gap: 4px !important;
                padding-right: 10px !important;
            }
            .dash-header .column-header-name {
                flex: 1 !important;
                text-align: left !important;
            }
            .dash-header .sort-arrow {
                flex-shrink: 0 !important;
            }
            .dash-spreadsheet td .cell-markdown {
                line-height: 1.2 !important;
                padding: 0 !important;
                margin: 0 !important;
            }
            .row-highlight td {
                background-color: #cfe2ff !important;
            }

            /* Wrap ag-Grid header text only at word boundaries, never mid-word or ellipsis */
            .ag-header-cell-text {
                word-break: normal !important;
                overflow-wrap: normal !important;
                white-space: normal !important;
                overflow: visible !important;
                text-overflow: clip !important;
            }
            /* Keep the sort arrow visible and un-squashed alongside wrapped header text */
            .ag-header-cell-label {
                flex-wrap: nowrap !important;
            }
            .ag-sort-indicator-container {
                flex-shrink: 0 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


# ── Layout components ─────────────────────────────────────

mytitle = dcc.Markdown(children='', style={'margin': '10px 0 0 0'})

mygraph = dcc.Graph(
    figure={},
    config={
        'scrollZoom': True,
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
    },
    style={'height': '85vh'}
)

# ── Static components ─────────────────────────────────────

top5_overall_component = render_top5_bars(
    petitions_list[petitions_list['status'] == 'open'],
    'total_signature_count',
    bar_color='#006548',
    border_color='#003f2d'
)

# ── Dropdowns ─────────────────────────────────────────────

petition_options = petitions_list[['petition_id', 'petition_title']].copy()

petition_dropdown = dbc.RadioItems(
    id='petition-dropdown',
    options=[
        {'label': row['petition_title'], 'value': row['petition_id']}
        for _, row in petition_options.iterrows()
    ],
    value=petition_options.iloc[0]['petition_id'],
    style={'maxHeight': '80vh', 'overflowY': 'auto', 'padding': '10px'}
)

constituency_dropdown = dcc.Dropdown(
    id='analytics-petition-dropdown',
    options=[
        {'label': row['constituency_name'], 'value': row['PCON24CD']}
        for _, row in pcon24cds.iterrows()
    ],
    value=pcon24cds.iloc[0]['PCON24CD'],
    clearable=False,
    style={'width': '260px'}
)

upcoming_debate_options = petitions_list[
    petitions_list['scheduled_debate_date'].notna() &
    (pd.to_datetime(petitions_list['scheduled_debate_date']) >= pd.Timestamp.now())
][['petition_id', 'petition_title', 'scheduled_debate_date']].drop_duplicates().sort_values('scheduled_debate_date')

upcoming_debate_dropdown = dcc.Dropdown(
    id='upcoming-debate-dropdown',
    options=[
        {
            'label': f"{row['petition_title']} ({pd.to_datetime(row['scheduled_debate_date']).strftime('%d %b %Y')})",
            'value': row['petition_id']
        }
        for _, row in upcoming_debate_options.iterrows()
    ],
    value=upcoming_debate_options.iloc[0]['petition_id'] if len(upcoming_debate_options) else None,
    clearable=False,
    style={'width': '380px'}
)

# ── Banner ─────────────────────────────────────────────────

page_nav = dbc.Nav([
    dbc.NavLink("Constituency Overview", id='tab-1-navlink', active=True),
    dbc.NavLink("Map", id='tab-2-navlink', active=False),
], pills=True)

banner = dbc.Navbar(
    dbc.Container([
        html.H3("UK Petitions Dashboard", className="text-white mb-0"),
        page_nav,
        dbc.Row([
            dbc.Col(html.Label("Constituency:", className="text-white mb-0 me-2"), width="auto"),
            dbc.Col(constituency_dropdown, width="auto"),
        ], align="center", className="g-2 flex-nowrap"),
    ], fluid=True, style={'paddingRight': '32px'}),
    color="primary",
    dark=True,
)


# ── App layout ────────────────────────────────────────────

app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    banner,
    dbc.Container([
        dcc.Tabs(id='main-tabs', value='tab-1', children=[

            dcc.Tab(value='tab-1', children=[
                html.Div([

                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5("Top 5 petitions overall (all constituencies)", className="mb-1 text-center"),
                                    top5_overall_component
                                ], className="pt-2 pb-2"),
                                className="shadow-sm h-100"
                            )
                        ], md=6),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5(id='top-5-raw-title', className="mb-1 text-center"),
                                    html.Div(id='top-5-table-raw-count')
                                ], className="pt-2 pb-2"),
                                className="shadow-sm h-100"
                            )
                        ], md=6)
                    ], className="g-2"),

                    # ── Upcoming debates ────────────────────────────────────
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(
                                            html.H5("Upcoming Debates: signatures by constituency", className="mb-1"),
                                            width="auto"
                                        ),
                                        dbc.Col(
                                            dbc.Row([
                                                dbc.Col(html.Label("Select petition:", className="fw-bold mb-0 me-2"), width="auto"),
                                                dbc.Col(upcoming_debate_dropdown, width="auto"),
                                            ], align="center", className="g-2 flex-nowrap"),
                                            width="auto"
                                        ),
                                    ], align="center", justify="between", className="mb-2 g-0"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardBody([
                                                    html.H6("Debate date", className="text-muted mb-1", style={'fontSize': '12px'}),
                                                    html.H5(id='debate-date-box', className="mb-0")
                                                ])
                                            ], className="mb-2 shadow-sm", style={'borderRadius': '10px'}),
                                            dbc.Card([
                                                dbc.CardBody([
                                                    html.H6("Total votes (all constituencies)", className="text-muted mb-1", style={'fontSize': '12px'}),
                                                    html.H5(id='debate-total-votes-box', className="mb-0")
                                                ])
                                            ], className="mb-2 shadow-sm", style={'borderRadius': '10px'}),
                                            dbc.Card([
                                                dbc.CardBody([
                                                    html.H6("Votes in selected constituency", className="text-muted mb-1", style={'fontSize': '12px'}),
                                                    html.H5(id='debate-constituency-votes-box', className="mb-0")
                                                ])
                                            ], className="shadow-sm", style={'borderRadius': '10px'}),
                                        ], md=3),
                                        dbc.Col([
                                            dcc.Graph(id='upcoming-debates-histogram', style={'height': '350px'})
                                        ], md=9)
                                    ])
                                ], className="pt-2 pb-2"),
                                className="shadow-sm"
                            )
                        ])
                    ], className="mt-4"),

                    # ── All open petitions table ───────────────────────────
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5("All Petitions", className="mb-3 text-center"),
                                    html.Div(id='all-petitions-table')
                                ], className="pt-2 pb-2"),
                                className="shadow-sm"
                            )
                        ])
                    ], className="mt-4")

                ], style={'padding': '20px'})
            ]),

            dcc.Tab(value='tab-2', children=[
                dbc.Row([
                    dbc.Col([
                        html.H5("Select a Petition", className="mb-3"),
                        petition_dropdown
                    ], style={
                        'backgroundColor': '#f8f9fa',
                        'padding': '20px',
                        'borderRight': '2px solid #dee2e6',
                        'minWidth': '400px',
                        'maxWidth': '400px'
                    }),
                    dbc.Col([
                        dbc.Row([mytitle], className="g-0", style={'marginBottom': '0'}),
                        dbc.Row([
                            dbc.Col([
                                dcc.Loading(
                                    id="loading",
                                    type="circle",
                                    children=[mygraph]
                                )
                            ], width=8),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("Total Signatures", className="text-white mb-2"),
                                        html.H3(id='total-sigs', className="mb-0 text-white")
                                    ])
                                ], className="mb-3 shadow", color="primary", inverse=True,
                                   style={'borderRadius': '15px'}),
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("Scheduled debate date", className="text-white mb-2"),
                                        html.H3(id='sch-debate-date', className="mb-0 text-white")
                                    ])
                                ], className="mb-3 shadow", color="info", inverse=True,
                                   style={'borderRadius': '15px'}),
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("Constituency with most signatures", className="text-white mb-2"),
                                        html.H3(id='highest-count-con', className="mb-0 text-white")
                                    ])
                                ], className="mb-3 shadow", color="success", inverse=True,
                                   style={'borderRadius': '15px'}),
                            ], width=3)
                        ])
                    ], style={'flex': '1'})
                ], style={'minHeight': '80vh'})
            ])

        ])
    ], fluid=True)
])


#####################
#### Callbacks   ####
#####################

# ── Top nav (drives the hidden dcc.Tabs) ──────────────────

@app.callback(
    Output('main-tabs', 'value'),
    Output('tab-1-navlink', 'active'),
    Output('tab-2-navlink', 'active'),
    Input('tab-1-navlink', 'n_clicks'),
    Input('tab-2-navlink', 'n_clicks'),
    prevent_initial_call=True
)
def switch_tab(_n1, _n2):
    if ctx.triggered_id == 'tab-2-navlink':
        return 'tab-2', False, True
    return 'tab-1', True, False


# ── Constituency Overview tab ─────────────────────────────

@app.callback(
    Output('top-5-raw-title', 'children'),
    Output('top-5-table-raw-count', 'children'),
    Input('analytics-petition-dropdown', 'value')
)
def update_top5_raw(PCON24CD):
    open_df = petitions_df[
        (petitions_df['PCON24CD'] == PCON24CD) &
        (petitions_df['status'] == 'open')
    ].copy()

    constituency_name = pcon24cds.loc[pcon24cds['PCON24CD'] == PCON24CD, 'constituency_name'].iloc[0]
    title = f"Top 5 petitions in {constituency_name}"

    return title, render_top5_bars(
        open_df, 'signature_count',
        bar_color='#40a583', border_color='#1a7a5c',
        secondary_col='total_signature_count',
        marker_col='median_signature_count'
    )


@app.callback(
    Output('debate-date-box', 'children'),
    Output('debate-total-votes-box', 'children'),
    Output('debate-constituency-votes-box', 'children'),
    Output('upcoming-debates-histogram', 'figure'),
    Input('upcoming-debate-dropdown', 'value'),
    Input('analytics-petition-dropdown', 'value')
)
def update_debate_section(petition_id, PCON24CD):
    df = petitions_df[petitions_df['petition_id'] == petition_id]

    debate_date = df['scheduled_debate_date'].iloc[0]
    debate_date_str = pd.to_datetime(debate_date).strftime('%d %b %Y') if pd.notna(debate_date) else 'Not scheduled'

    total_votes = df['total_signature_count'].iloc[0]

    constituency_votes = df.loc[df['PCON24CD'] == PCON24CD, 'signature_count'].iloc[0]

    counts, bin_edges = np.histogram(df['signature_count'], bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    constituency_bin = np.searchsorted(bin_edges, constituency_votes, side='right') - 1
    constituency_bin = min(max(constituency_bin, 0), len(counts) - 1)

    bar_colors = ['#0d6efd'] * len(counts)
    bar_colors[constituency_bin] = '#fd7e14'

    fig = go.Figure(go.Bar(x=bin_centers, y=counts, width=bin_widths, marker_color=bar_colors))
    fig.update_layout(
        xaxis_title='Signatures in constituency',
        yaxis_title='Number of constituencies',
        margin={'r': 20, 't': 20, 'l': 60, 'b': 40},
        bargap=0.05
    )

    return debate_date_str, f"{total_votes:,}", f"{constituency_votes:,}", fig


# ── All Petitions table ───────────────────────────────────

@app.callback(
    Output('all-petitions-table', 'children'),
    Input('analytics-petition-dropdown', 'value'),
)
def update_all_petitions_table(PCON24CD):
    today = datetime.now().date()

    # Constituency-level stats for the selected constituency
    constituency_sigs = petitions_df[petitions_df['PCON24CD'] == PCON24CD][[
        'petition_id', 'signature_count',
        'percentile_rank_raw', 'percentile_rank_pop',
        'sig_per_pop', 'sig_per_pop_rank'
    ]].drop_duplicates(subset='petition_id').copy()
    constituency_sigs['petition_id'] = constituency_sigs['petition_id'].astype(str)

    df = petitions_list.copy()
    df['petition_id'] = df['petition_id'].astype(str)
    df = df.merge(constituency_sigs, on='petition_id', how='left')

    # Days open
    df['opened_at'] = pd.to_datetime(df['opened_at']).dt.date
    df['months_open'] = df['opened_at'].apply(lambda d: (today - d).days).apply(
        lambda n: 'Less than 1 month' if n < 30 else '1-3 months' if n < 90 else '4-6 months' if n < 180 else '6+ months'
    )

    # Debate date formatting
    df['scheduled_debate_date'] = pd.to_datetime(df['scheduled_debate_date'], errors='coerce').dt.date

    # Clickable title as markdown
    df['petition_title_link'] = df.apply(
        lambda r: f"[{r['petition_title']}]({r['petition_url']})", axis=1
    )

    df['signature_count'] = df['signature_count'].fillna(0).astype(int)
    df['percentile_rank_raw'] = df['percentile_rank_raw'].round(1)
    df['percentile_rank_pop'] = df['percentile_rank_pop'].round(1)
    df['sig_per_pop'] = df['sig_per_pop'].round(2)
    df['sig_per_pop_rank'] = df['sig_per_pop_rank'].fillna(0).astype(int)

    table_df = df[[
        'petition_title_link',
        'opened_at',
        'months_open',
        'total_signature_count',
        'signature_count',
        'percentile_rank_raw',
        'sig_per_pop',
        'sig_per_pop_rank',
        'percentile_rank_pop',
        'scheduled_debate_date'
    ]].sort_values('total_signature_count', ascending=False)

    number_format = {'function': "d3.format(',')(params.value)"}

    return dag.AgGrid(
        id='all-petitions-datatable',
        rowData=table_df.to_dict('records'),
        columnDefs=[
            {'field': 'petition_title_link', 'headerName': 'Petition', 'cellRenderer': 'markdown',
             'filter': 'agTextColumnFilter',
             'filterParams': {'filterOptions': ['contains', 'notContains']},
             'cellClass': 'petition-title-cell',
             'sortable': False,
             'flex': 1.6, 'minWidth': 220, 'wrapText': True, 'autoHeight': True},
            {'field': 'opened_at', 'headerName': 'Date opened', 'flex': 0.9, 'minWidth': 125},
            {'field': 'months_open', 'headerName': 'Months open', 'flex': 0.8, 'minWidth': 110},
            {'field': 'total_signature_count', 'headerName': 'Total signatures',
                'valueFormatter': number_format, 'flex': 1, 'minWidth': 130},
            {
                'headerName': 'Raw counts',
                'children': [
                    {'field': 'signature_count', 'headerName': 'Constituency signatures',
                     'valueFormatter': number_format, 'flex': 1, 'minWidth': 150},
                    {'field': 'percentile_rank_raw', 'headerName': 'Percentile ranking (counts)', 'flex': 1, 'minWidth': 130},
                ]
            },
            {
                'headerName': 'Counts per 1,000 population',
                'children': [
                    {'field': 'sig_per_pop', 'headerName': 'Signatures per 1,000', 'flex': 1, 'minWidth': 130},
                    {'field': 'percentile_rank_pop', 'headerName': 'Percentile ranking (sig per 1000)', 'flex': 1, 'minWidth': 130},
                ]
            },
            {'field': 'scheduled_debate_date', 'headerName': 'Scheduled debate', 'flex': 0.8, 'minWidth': 135},
        ],
        defaultColDef={'sortable': True, 'resizable': True, 'wrapHeaderText': True, 'autoHeaderHeight': True},
        dashGridOptions={'pagination': True, 'paginationPageSize': 20, 'domLayout': 'autoHeight', 'unSortIcon': True},
        dangerously_allow_code=True,
        className='ag-theme-alpine',
        style={'width': '100%'},
    )


# ── Map tab ───────────────────────────────────────────────

@app.callback(
    Output(mygraph, 'figure'),
    Output(mytitle, 'children'),
    Output('total-sigs', 'children'),
    Output('sch-debate-date', 'children'),
    Output('highest-count-con', 'children'),
    Input('petition-dropdown', 'value')
)
def update_graph(petition_id):
    callback_start = time.time()

    cached_data = get_petition_data(petition_id)

    df = pd.DataFrame(cached_data, columns=petitions_df.columns)
    print(f"Time to retrieve and process petition data: {time.time() - callback_start:.4f}s")

    petition_title = df['petition_title'].iloc[0] if len(df) > 0 else "No data"
    total_signatures = df['signature_count'].sum()

    sch_debate_date = df['scheduled_debate_date'].iloc[0] if 'scheduled_debate_date' in df.columns else None
    debate_date_str = str(sch_debate_date) if pd.notna(sch_debate_date) else "Not scheduled"

    max_row = df.loc[df['signature_count'].idxmax()]
    highest_count_con = max_row['constituency_name']
    highest_count = max_row['signature_count']

    max_color = petition_quantiles.get(petition_id, df['signature_count'].max())

    plot_start = time.time()
    fig = px.choropleth(
        df,
        locations='PCON24CD',
        geojson=constituency_boundaries,
        featureidkey="properties.PCON24CD",
        color='signature_count',
        color_continuous_scale="Viridis",
        range_color=[0, max_color],
        labels={
            'signature_count': 'Number of signatures',
            'PCON24CD': 'Constituency Code',
            'constituency_name': 'Constituency'
        },
        hover_data={
            'PCON24CD': False,
            'constituency_name': True,
            'signature_count': True
        }
    )

    fig.update_geos(
        visible=False,
        projection_scale=0.8,
        center=dict(lat=54.5, lon=-3),
        lataxis_range=[48, 60],
        lonaxis_range=[-10, 4]
    )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        uirevision='constant',
        coloraxis_colorbar=dict(
            title="Signatures",
            thickness=15,
            len=0.7,
            x=1.0,
            xanchor="left",
            y=0.5,
            yanchor="middle"
        )
    )

    print(f"Time to plot the choropleth: {time.time() - plot_start:.4f}s")

    return (
        fig,
        '# ' + petition_title,
        f"{total_signatures:,}",
        debate_date_str,
        f"{highest_count_con} ({highest_count:,})"
    )


if __name__ == '__main__':
    app.run(debug=(ENV == 'local'), port=8051)