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
import geopandas as gpd
import plotly.express as px
from dash import Dash, dcc, Output, Input, html, dash_table, no_update
from dash.dash_table.Format import Format, Group
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
        constituencies = gpd.read_file(local_path)
    else:
        constituencies = load_geojson(
            'static data/constituencies_july_2024.geojson'
        )
    constituencies = constituencies[['PCON24CD', 'geometry']]
    constituencies['geometry'] = constituencies['geometry'].simplify(0.005)
    return constituencies


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
constituencies = get_constituency_geojson()

print("Loading petitions data...")
petitions_list, petitions_count = get_petitions_data()

print("Loading population data...")
pop_df = get_population_data()

print("Done loading data.")

# Adding data on population
petitions_count = petitions_count.merge(pop_df[['PCON24CD', 'pop']], on='PCON24CD', how='left')

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
            .tab {
                padding: 12px 24px !important;
                border: none !important;
                background-color: #e9ecef !important;
            }
            .tab--selected {
                background-color: white !important;
            }
            .tabs {
                border-bottom: 1px solid #dee2e6 !important;
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

banner = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H3("UK Petitions Dashboard", className="text-white mb-0")),
        ], align="center", className="g-0"),
    ], fluid=True),
    color="primary",
    dark=True,
)

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

days_open_checklist = dcc.Checklist(
    id='days-open-filter',
    options=[
        {'label': 'Less than 1 month', 'value': 'Less than 1 month'},
        {'label': '1-3 months', 'value': '1-3 months'},
        {'label': '4-6 months', 'value': '4-6 months'},
        {'label': '6+ months', 'value': '6+ months'},
    ],
    value=['Less than 1 month', '1-3 months', '4-6 months', '6+ months'],
    inline=True
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

constituency_options = petitions_df[['PCON24CD', 'constituency_name']].drop_duplicates()

constituency_dropdown = dcc.Dropdown(
    id='analytics-petition-dropdown',
    options=[
        {'label': row['constituency_name'], 'value': row['PCON24CD']}
        for _, row in constituency_options.iterrows()
    ],
    value=constituency_options.iloc[0]['PCON24CD'],
    clearable=False
)


# ── App layout ────────────────────────────────────────────

app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    banner,
    dbc.Container([
        dcc.Tabs(id='main-tabs', value='tab-1', children=[

            dcc.Tab(label='Constituency Overview', value='tab-1', children=[
                html.Div([

                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select constituency:", className="fw-bold mb-2"),
                                    constituency_dropdown
                                ], md=4)
                            ])
                        ])
                    ], className="mb-4"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5("Top 5 by raw count", className="mb-1 text-center"),
                                    html.Div(id='top-5-table-raw-count')
                                ], className="pt-2 pb-2"),
                                className="shadow-sm h-100"
                            )
                        ], md=6),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5("Upcoming Debates", className="mb-1 text-center"),
                                    html.Div(id='upcoming-debates-table')
                                ], className="pt-2 pb-2"),
                                className="shadow-sm h-100"
                            )
                        ], md=6)
                    ]),

                    # ── All Petitions table ───────────────────────────
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5("All Petitions", className="mb-3 text-center"),
                                    days_open_checklist,
                                    html.Div(id='all-petitions-table')
                                ], className="pt-2 pb-2"),
                                className="shadow-sm"
                            )
                        ])
                    ], className="mt-4")

                ], style={'padding': '20px'})
            ]),

            dcc.Tab(label='Map', value='tab-2', children=[
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

# ── Constituency Overview tab ─────────────────────────────

@app.callback(
    Output('top-5-table-raw-count', 'children'),
    Input('analytics-petition-dropdown', 'value')
)
def update_top5_raw(PCON24CD):
    open_df = petitions_df[
        (petitions_df['PCON24CD'] == PCON24CD) &
        (petitions_df['status'] == 'open')
    ].copy()

    top_5 = open_df.nlargest(5, 'signature_count')[
        ['petition_title', 'signature_count', 'petition_url']
    ].reset_index(drop=True)

    max_val = top_5['signature_count'].max()
    padding = max_val * 0.15
    top_5['petition_title_wrapped'] = top_5['petition_title'].apply(wrap_text)

    fig = px.bar(
        top_5,
        x='signature_count',
        y='petition_title_wrapped',
        orientation='h',
        custom_data=['petition_url'],
        labels={'signature_count': 'Signatures', 'petition_title_wrapped': ''}
    )

    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=300,
        margin=dict(l=150, r=20, t=10, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.5
    )

    fig.update_traces(
        marker_color='#0d6efd',
        marker_line_color='#0a58ca',
        marker_line_width=1.5,
        opacity=0.8,
        text=top_5['signature_count'].apply(lambda x: f'{x:,}'),
        textposition='outside',
        textfont=dict(size=11, color='#333'),
        hovertemplate=None,
        hoverinfo='none'
    )

    fig.update_xaxes(
        range=[-150, max_val + padding],
        tickformat=',',
        showgrid=True,
        gridcolor='#e9ecef'
    )

    fig.update_yaxes(ticklen=40)

    return dcc.Graph(
        id='top-5-bar-chart-raw-count',
        figure=fig,
        config={'displayModeBar': False},
        hoverData=None
    )


@app.callback(
    Output('url', 'href'),
    Input('top-5-bar-chart-raw-count', 'clickData'),
    prevent_initial_call=True
)
def redirect_on_bar_click(click_raw):
    if not click_raw or 'points' not in click_raw:
        return no_update
    return click_raw['points'][0]['customdata'][0]


@app.callback(
    Output('upcoming-debates-table', 'children'),
    Input('analytics-petition-dropdown', 'value')
)
def update_scheduled_debates(PCON24CD):
    df = petitions_df[
            (petitions_df['PCON24CD'] == PCON24CD) &
            (petitions_df['scheduled_debate_date'].notna()) &
            (pd.to_datetime(petitions_df['scheduled_debate_date']) >= pd.Timestamp.now())
        ][['petition_title', 'scheduled_debate_date', 'signature_count', 'median_signature_count',
           'sig_per_pop', 'sig_per_pop_rank']].drop_duplicates().sort_values('scheduled_debate_date').head(5).copy()

    df['scheduled_debate_date'] = pd.to_datetime(df['scheduled_debate_date']).dt.strftime('%d %b %Y')
    df['sig_per_pop'] = df['sig_per_pop'].round(2)

    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[
            {'name': 'Petition', 'id': 'petition_title'},
            {'name': 'Debate Date', 'id': 'scheduled_debate_date'},
            {'name': 'No. of sigs', 'id': 'signature_count'},
            {'name': 'Median sig count', 'id': 'median_signature_count'},
            {'name': 'Sigs per 1,000', 'id': 'sig_per_pop'},
            {'name': 'Rank (sigs per 1,000)', 'id': 'sig_per_pop_rank'},
        ],
        style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '13px',
                    'fontFamily': 'sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
        style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold', 'borderBottom': '2px solid #dee2e6'},
        style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}]
    )


# ── All Petitions table ───────────────────────────────────

@app.callback(
    Output('all-petitions-table', 'children'),
    Input('analytics-petition-dropdown', 'value'),
    Input('days-open-filter', 'value'),
)
def update_all_petitions_table(PCON24CD, days_open_selected):
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
    df['days_open_interval'] = df['opened_at'].apply(lambda d: (today - d).days).apply(
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
        'petition_title_link', 'opened_at',
        'days_open_interval',
        'total_signature_count',
        'signature_count',
        'percentile_rank_raw',
        'percentile_rank_pop',
        'sig_per_pop',
        'sig_per_pop_rank',
        'scheduled_debate_date'
    ]].sort_values('total_signature_count', ascending=False)

    table_df = table_df[table_df['days_open_interval'].isin(days_open_selected)]

    return dash_table.DataTable(
        id='all-petitions-datatable',
        cell_selectable=False,
        data=table_df.to_dict('records'),
        columns=[
            {'name': ['', 'Petition'], 'id': 'petition_title_link', 'presentation': 'markdown'},
            {'name': ['', 'Date opened'], 'id': 'opened_at', 'type': 'datetime'},
            {'name': ['', 'Days open'], 'id': 'days_open_interval'},
            {'name': ['All petitions', 'Total signatures'], 'id': 'total_signature_count', 'type': 'numeric', 'format': Format(group=Group.yes)},
            {'name': ['Raw counts', 'Constituency signatures'], 'id': 'signature_count', 'type': 'numeric', 'format': Format(group=Group.yes)},
            {'name': ['Raw counts', 'Percentile ranking (counts)'], 'id': 'percentile_rank_raw', 'type': 'numeric'},
            {'name': ['Counts per 1,000 population', 'Signatures per 1,000'], 'id': 'sig_per_pop', 'type': 'numeric'},
            {'name': ['Counts per 1,000 population', 'Percentile ranking (sig per 1000)'], 'id': 'percentile_rank_pop', 'type': 'numeric'},
            {'name': ['', 'Scheduled debate'], 'id': 'scheduled_debate_date', 'type': 'datetime'},
        ],
        merge_duplicate_headers=True,
        sort_action='native',
        sort_mode='single',
        page_action='native',
        page_size=20,
        style_cell={
            'textAlign': 'left',
            'padding': '4px 12px',
            'fontSize': '13px',
            'fontFamily': 'sans-serif',
            'whiteSpace': 'normal',
            'height': 'auto',
            'overflow': 'hidden',
            'boxSizing': 'border-box',
            'verticalAlign': 'top',
        },
        style_cell_conditional=[
            # Fixed widths on every column so nothing shifts on sort or content change
            {'if': {'column_id': 'petition_title_link'},       'width': '300px', 'minWidth': '300px', 'maxWidth': '300px', 'whiteSpace': 'normal'},
            {'if': {'column_id': 'opened_at'},                 'width': '100px',  'minWidth': '100px',  'maxWidth': '100px',  'textAlign': 'right'},
            {'if': {'column_id': 'days_open_interval'}, 'width': '120px', 'minWidth': '120px', 'maxWidth': '120px', 'textAlign': 'right'},
            {'if': {'column_id': 'total_signature_count'},     'width': '100px', 'minWidth': '100px', 'maxWidth': '100px', 'textAlign': 'right'},
            {'if': {'column_id': 'signature_count'},           'width': '130px', 'minWidth': '130px', 'maxWidth': '130px', 'textAlign': 'right'},
            {'if': {'column_id': 'percentile_rank_raw'},           'width': '110px', 'minWidth': '110px', 'maxWidth': '110px', 'textAlign': 'right'},
            {'if': {'column_id': 'sig_per_pop'},               'width': '110px', 'minWidth': '110px', 'maxWidth': '110px', 'textAlign': 'right'},
            {'if': {'column_id': 'percentile_rank_pop'},          'width': '110px', 'minWidth': '110px', 'maxWidth': '110px', 'textAlign': 'right'},
            {'if': {'column_id': 'scheduled_debate_date'},     'width': '120px', 'minWidth': '120px', 'maxWidth': '120px', 'textAlign': 'right'},
        ],
        style_header={
            'backgroundColor': '#f8f9fa',
            'fontWeight': 'bold',
            'borderBottom': '2px solid #dee2e6',
            'textAlign': 'left',
            'padding': '4px 12px',
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'}
        ],
        style_table={
            'overflowX': 'auto',
            'tableLayout': 'fixed',
        }
    )


app.clientside_callback(
    """
    function(data) {
        setTimeout(() => {
            const rows = document.querySelectorAll('#all-petitions-datatable .dash-spreadsheet-container .dash-spreadsheet tbody tr');
            rows.forEach(row => {
                row.onclick = () => {
                    rows.forEach(r => r.classList.remove('row-highlight'));
                    row.classList.add('row-highlight');
                };
            });
        }, 100);
        return window.dash_clientside.no_update;
    }
    """,
    Output('all-petitions-datatable', 'style'),
    Input('all-petitions-datatable', 'data')
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
        geojson=constituencies,
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