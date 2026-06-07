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
import dash_bootstrap_components as dbc
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv


# ── Environment & AWS setup ───────────────────────────────

script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_DEFAULT_REGION')

bucket = 'uk-petitions-dashboard'

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
    constituencies = load_geojson(
        'static data/Westminster_Parliamentary_Constituencies_July_2024_Boundaries_UK_BGC_-8097874740651686118.geojson'
    )
    constituencies = constituencies[['PCON24CD', 'geometry']]
    constituencies['geometry'] = constituencies['geometry'].simplify(0.005)
    return constituencies


def get_petitions_data():
    petitions_list = load_csv('dynamic data/dashboard_list.csv')
    petitions_count = load_csv('dynamic data/dashboard_counts.csv')
    return petitions_list, petitions_count


@lru_cache(maxsize=128)
def get_petition_data(petition_id):
    start = time.time()
    df = petitions_df[petitions_df['petition_id'] == petition_id]
    print(f"Time to retrieve petition data from cache: {time.time() - start:.4f}s")
    return tuple(df.itertuples(index=False))


# ── Data processing ───────────────────────────────────────
print("Loading GeoJSON...")
constituencies = get_constituency_geojson()

print("Loading petitions data...")
petitions_list, petitions_count = get_petitions_data()

print("Done loading data.")

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

app = Dash(__name__, external_stylesheets=[dbc.themes.PULSE])
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
                                    html.H5("Top 10 Petitions", className="mb-1 text-center"),
                                    html.Div(id='top-10-table')
                                ], className="pt-2 pb-2"),
                                className="shadow-sm"
                            )
                        ], md=6)
                    ])
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
    Output('top-10-table', 'children'),
    Input('analytics-petition-dropdown', 'value')
)
def update_kpi_cards(PCON24CD):
    open_df = petitions_df[
        (petitions_df['PCON24CD'] == PCON24CD) &
        (petitions_df['status'] == 'open')
    ].copy()

    top_10 = open_df.nlargest(10, 'signature_count')[
        ['petition_title', 'signature_count', 'petition_url']
    ].reset_index(drop=True)

    max_val = top_10['signature_count'].max()
    padding = max_val * 0.15

    top_10['petition_title_wrapped'] = top_10['petition_title'].apply(wrap_text)

    fig = px.bar(
        top_10,
        x='signature_count',
        y='petition_title_wrapped',
        orientation='h',
        custom_data=['petition_url'],
        labels={'signature_count': 'Signatures', 'petition_title_wrapped': ''}
    )

    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=600,
        margin=dict(l=280, r=20, t=10, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        bargap=0.5
    )

    fig.update_traces(
        marker_color='#0d6efd',
        marker_line_color='#0a58ca',
        marker_line_width=1.5,
        opacity=0.8,
        text=top_10['signature_count'].apply(lambda x: f'{x:,}'),
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
        id='top-10-bar-chart',
        figure=fig,
        config={'displayModeBar': False},
        hoverData=None
    )


@app.callback(
    Output('url', 'href'),
    Input('top-10-bar-chart', 'clickData'),
    prevent_initial_call=True
)
def redirect_on_bar_click(clickData):
    if not clickData or 'points' not in clickData:
        return no_update
    return clickData['points'][0]['customdata'][0]


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
    app.run(debug=False, port=8051)