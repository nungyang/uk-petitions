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
TOTAL_CONSTITUENCIES = len(pcon24cds)

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
petitions_count['sig_rank_raw'] = petitions_count.groupby('petition_id')['signature_count'].rank(ascending=False, method='min')
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

# Displays a percentile ranking (0 = best) as a "Top N%" / "Bottom 50%" category
# instead of the raw number, while the underlying cell value (used for sorting)
# stays numeric.
PERCENTILE_CATEGORY_FORMAT = {'function': (
    "params.value == null ? '' : "
    "params.value <= 1 ? 'Top 1%' : "
    "params.value <= 5 ? 'Top 5%' : "
    "params.value <= 10 ? 'Top 10%' : "
    "params.value <= 25 ? 'Top 25%' : "
    "params.value <= 50 ? 'Top 50%' : 'Bottom 50%'"
)}


def percentile_category(value):
    """Python equivalent of PERCENTILE_CATEGORY_FORMAT for use outside ag-Grid cells."""
    if value is None or pd.isna(value):
        return None
    if value <= 1:
        return 'Top 1%'
    if value <= 5:
        return 'Top 5%'
    if value <= 10:
        return 'Top 10%'
    if value <= 25:
        return 'Top 25%'
    if value <= 50:
        return 'Top 50%'
    return 'Bottom 50%'


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
            'borderLeft': '3px solid #D55E00'
        }))

    return html.Div(
        track_children, style={'position': 'relative', 'marginTop': '2px'}
    )


def render_top5_bars(df, value_col, bar_color='#0d6efd', border_color='#0a58ca', marker_col=None):
    """Render a top-5 horizontal bar list, bars scaled to value_col's max across the top 5,
    with the value and "signatures" shown below each bar. If marker_col is given, each bar
    also gets its own dotted vertical marker at that row's marker_col value (e.g. the
    petition's median signature count across all constituencies).

    Every row has the same fixed height (title area + bar + value block) so that rows line up
    across two side-by-side charts built from this function.
    """
    cols = ['petition_title', value_col, 'petition_url']
    if marker_col:
        cols.append(marker_col)

    top_5 = df.nlargest(5, value_col)[cols].sort_values(value_col, ascending=False).reset_index(drop=True)

    max_val = top_5[value_col].max()

    rows = []
    for _, row in top_5.iterrows():
        marker_pct = (row[marker_col] / max_val) * 100 if marker_col and max_val else None

        children = [
            html.Div(
                html.A(
                    row['petition_title'],
                    href=row['petition_url'],
                    target='_blank',
                    style={
                        'fontSize': '14px', 'color': '#333', 'textDecoration': 'none',
                        'display': '-webkit-box', 'WebkitLineClamp': '2', 'WebkitBoxOrient': 'vertical',
                        'overflow': 'hidden', 'textOverflow': 'ellipsis', 'width': '100%',
                        'lineHeight': '18px', 'maxHeight': '36px'
                    }
                ),
                style={'height': '36px', 'display': 'flex', 'alignItems': 'flex-end'}
            ),
            _render_bar(row[value_col], max_val, bar_color, border_color, marker_pct),
            html.Div([
                html.Span(f"{row[value_col]:,}", style={'fontWeight': 'bold', 'color': '#333'}),
                html.Span(" signatures", style={'color': '#777'})
            ], style={'fontSize': '11px', 'marginTop': '3px'})
        ]

        rows.append(html.Div(children, style={'marginBottom': '10px'}))

    return html.Div(rows, style={'padding': '10px 20px'})


def render_signature_histogram(df, median_value, highlight_value=None):
    """Histogram of signature_count across constituencies for one petition, with a
    dotted vertical line at median_value and, if highlight_value is given, the bin
    containing it picked out in a lighter shade of green.
    """
    max_val = int(df['signature_count'].max())
    bin_width = max(1, -(-max_val // 30))  # ceil(max_val / 30), so bin edges land on whole numbers
    num_bins = max(1, -(-max_val // bin_width))  # ceil(max_val / bin_width)
    bin_edges = np.arange(num_bins + 1) * bin_width
    counts, bin_edges = np.histogram(df['signature_count'], bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bin_labels = [
        f"{int(round(bin_edges[i]))} < no of sig ≤ {int(round(bin_edges[i + 1]))}"
        for i in range(len(bin_edges) - 1)
    ]

    bar_colors = ['#006548'] * len(counts)
    if highlight_value is not None:
        highlight_bin = np.searchsorted(bin_edges, highlight_value, side='right') - 1
        highlight_bin = min(max(highlight_bin, 0), len(counts) - 1)
        bar_colors[highlight_bin] = '#40a583'

    fig = go.Figure(go.Bar(
        x=bin_centers, y=counts, width=bin_widths, marker_color=bar_colors,
        text=bin_labels, textposition='none',
        hovertemplate='&nbsp;<br>&nbsp; &nbsp;%{y} constituencies &nbsp; &nbsp;<br>&nbsp; &nbsp;%{text} &nbsp; &nbsp;<br>&nbsp;<extra></extra>',
        hoverlabel=dict(bgcolor='white', font_color='#333')
    ))
    fig.update_layout(
        hovermode='x',
        xaxis_title='Number of signatures',
        yaxis_title='Number of constituencies',
        margin={'r': 20, 't': 20, 'l': 60, 'b': 40},
        bargap=0.05,
        font=dict(
            family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
            size=12, color='#333'
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(gridcolor='#e9ecef', zerolinecolor='#dee2e6', title_font=dict(size=15, color='#555'), tickfont=dict(size=13), fixedrange=True, showspikes=False,
                   tick0=0, dtick=max(1, round(max_val / 6)), tickformat=',d'),
        yaxis=dict(gridcolor='#e9ecef', zerolinecolor='#dee2e6', title_font=dict(size=15, color='#555'), tickfont=dict(size=13), fixedrange=True, showspikes=False),
    )
    fig.add_vline(x=median_value, line_width=2, line_color='#D55E00')

    return fig


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
            /* Smaller card title headings across the dashboard */
            .card-body h5 {
                font-size: 1rem;
            }

            /* Tighten the markdown-rendered two-line "signatures" cell: it inherits
               ag-Grid's row-height-driven line-height (~39px), which spaces the two
               lines out far more than a normal wrapped line, and center it. */
            .sig-ratio-cell {
                text-align: center;
                display: flex !important;
                align-items: flex-start;
                justify-content: center;
                padding-top: 8px;
            }
            .sig-ratio-cell .agGrid-Markdown div {
                line-height: 1.3;
            }

            /* Same treatment as .sig-ratio-cell, but with extra space between the
               ranking value and its "(Top N%)" category line. */
            .rank-ratio-cell {
                text-align: center;
                display: flex !important;
                align-items: flex-start;
                justify-content: center;
                padding-top: 8px;
            }
            .rank-ratio-cell .agGrid-Markdown div {
                line-height: 1.75;
            }

            /* AG-Grid's own CSS gives .ag-center-cols-viewport a min-height: 100%, which
               (in domLayout=autoHeight mode, where the viewport's own height is derived
               from its content) creates a circular/self-inflating height once wrapText
               row heights come in shorter than the grid's initial estimate — leaving a
               large empty gap below the last row. Break that feedback loop here. */
            #top5-percent-datatable .ag-center-cols-viewport,
            #top5-percent-datatable .ag-center-cols-container {
                min-height: unset !important;
            }

            /* hovermode='x' on the histogram shows a small floating axis-value label
               (e.g. "514.98") near the x-axis on hover, independent of showspikes; hide it. */
            #upcoming-debates-histogram .axistext,
            #petition-histogram .axistext {
                display: none !important;
            }

            /* Center the "Ranking"/"Signatures" column headers in the top 5% table */
            .ag-header-center .ag-header-cell-label {
                justify-content: center;
            }
            .ag-header-center .ag-header-cell-text {
                text-align: center;
            }

            /* Constituency dropdown placeholder */
            #analytics-petition-dropdown .Select-placeholder {
                text-align: left;
                font-size: 15px;
            }

            /* When reopened with a value already selected, hide the pre-filled value
               label so the search box reads as empty (JS below injects a real
               placeholder onto the input in this state). */
            #analytics-petition-dropdown .Select.is-open.has-value .Select-value,
            #debate-date-dropdown .Select.is-open.has-value .Select-value,
            #upcoming-debate-dropdown .Select.is-open.has-value .Select-value {
                display: none;
            }

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
        <script>
            // Some dropdowns hide their pre-filled value label on reopen (see CSS above);
            // this injects a real placeholder onto the now-empty search input for each one
            // so it reads e.g. "Search for constituency" instead of just being blank.
            (function() {
                var DROPDOWNS = [
                    {id: 'analytics-petition-dropdown', placeholder: 'Search for constituency', width: '210px'},
                    {id: 'debate-date-dropdown', placeholder: 'Select date', width: '150px'},
                    {id: 'upcoming-debate-dropdown', placeholder: 'Select petition', width: '110px'}
                ];
                var observer = new MutationObserver(function() {
                    DROPDOWNS.forEach(function(cfg) {
                        var wrapper = document.getElementById(cfg.id);
                        if (!wrapper) { return; }
                        var selectEl = wrapper.querySelector('.Select');
                        var input = wrapper.querySelector('.Select-input input');
                        if (!selectEl || !input) { return; }

                        var isOpenWithValue = selectEl.classList.contains('is-open') &&
                            selectEl.classList.contains('has-value');

                        // react-select continually re-sizes this input to match whatever was
                        // last typed (down to 5px once empty) and clips overflow, so the width
                        // must be re-applied every time this state is seen, not just once —
                        // otherwise a previous typed search leaves it too narrow next time.
                        if (isOpenWithValue) {
                            if (!input.placeholder) { input.placeholder = cfg.placeholder; }
                            if (input.value === '' && input.style.width !== cfg.width) {
                                input.style.width = cfg.width;
                            }
                        } else if (input.style.width || input.placeholder) {
                            // Closing without selecting (e.g. clicking away) leaves this input's
                            // widened, empty box — and its placeholder text — sitting on top of
                            // the value label that reappears, so clear both.
                            input.style.width = '';
                            input.placeholder = '';
                        }
                    });
                });
                observer.observe(document.body, {childList: true, subtree: true, attributes: true});
            })();
        </script>
    </body>
</html>
'''


# ── Layout components ─────────────────────────────────────

mygraph = dcc.Graph(
    figure={},
    config={
        'scrollZoom': True,
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
    },
    style={'height': '72vh'}
)

# ── Static components ─────────────────────────────────────

top5_overall_component = render_top5_bars(
    petitions_list[petitions_list['status'] == 'open'],
    'total_signature_count',
    bar_color='#006548',
    border_color='#003f2d'
)

_today = datetime.now().date()
_days_open = (_today - pd.to_datetime(petitions_list['opened_at'], dayfirst=True).dt.date).apply(lambda d: d.days)

up_and_coming_df = petitions_list[
    (petitions_list['status'] == 'open') &
    (_days_open < 30) &
    (petitions_list['total_signature_count'] >= 10000)
][['petition_title', 'petition_url', 'opened_at', 'total_signature_count']].sort_values(
    'total_signature_count', ascending=False
).copy()
up_and_coming_df['petition_title_link'] = up_and_coming_df.apply(
    lambda r: f"[{r['petition_title']}]({r['petition_url']})", axis=1
)
up_and_coming_df['days_open'] = _days_open.loc[up_and_coming_df.index] + 1
up_and_coming_df['avg_sig_per_day'] = (up_and_coming_df['total_signature_count'] / up_and_coming_df['days_open']).round(0)

if up_and_coming_df.empty:
    up_and_coming_component = html.Div(
        "No petitions opened in the last month have reached 10,000 signatures yet.",
        style={'padding': '10px 20px', 'color': '#777', 'fontSize': '13px'}
    )
else:
    up_and_coming_component = dag.AgGrid(
        id='up-and-coming-datatable',
        rowData=up_and_coming_df.to_dict('records'),
        columnDefs=[
            {'field': 'petition_title_link', 'headerName': 'Petition', 'cellRenderer': 'markdown',
             'cellClass': 'petition-title-cell', 'sortable': False,
             'flex': 2, 'minWidth': 180, 'wrapText': True, 'autoHeight': True},
            {'field': 'days_open', 'headerName': 'Days opened', 'flex': 0.6, 'minWidth': 88, 'sort': 'asc', 'sortable': False,
             'headerClass': 'ag-header-center', 'cellStyle': {'textAlign': 'center'}},
            {'field': 'total_signature_count', 'headerName': 'Total no. of sigs',
             'valueFormatter': {'function': "d3.format(',')(params.value)"}, 'flex': 0.7, 'minWidth': 92,
             'headerClass': 'ag-header-center', 'cellStyle': {'textAlign': 'center'}},
            {'field': 'avg_sig_per_day', 'headerName': 'Avg no. of sig per day',
             'valueFormatter': {'function': "d3.format(',')(params.value)"}, 'flex': 1, 'minWidth': 115,
             'headerClass': 'ag-header-center', 'cellStyle': {'textAlign': 'center'}},
        ],
        defaultColDef={'sortable': False, 'resizable': False, 'wrapHeaderText': True, 'autoHeaderHeight': True},
        dashGridOptions={'domLayout': 'autoHeight'},
        dangerously_allow_code=True,
        className='ag-theme-alpine',
        style={'width': '100%'},
    )

# ── Dropdowns ─────────────────────────────────────────────

petition_options = petitions_list[['petition_id', 'petition_title']].copy()

petition_dropdown = dcc.Dropdown(
    id='petition-dropdown',
    options=[
        {'label': row['petition_title'], 'value': row['petition_id']}
        for _, row in petition_options.iterrows()
    ],
    value=petition_options.iloc[0]['petition_id'],
    clearable=False,
    style={'width': '100%'}
)

constituency_dropdown = dcc.Dropdown(
    id='analytics-petition-dropdown',
    options=[
        {'label': row['constituency_name'], 'value': row['PCON24CD']}
        for _, row in pcon24cds.iterrows()
    ],
    placeholder='Select a constituency',
    clearable=False,
    style={'width': '320px'}
)

NO_CONSTITUENCY_MESSAGE = "Select a constituency from the dropdown (see top right)"

upcoming_debate_options = petitions_list[
    petitions_list['scheduled_debate_date'].notna() &
    (pd.to_datetime(petitions_list['scheduled_debate_date'], dayfirst=True) >= pd.Timestamp.now().normalize())
][['petition_id', 'petition_title', 'scheduled_debate_date']].drop_duplicates().sort_values('scheduled_debate_date')

upcoming_debate_dropdown = dcc.Dropdown(
    id='upcoming-debate-dropdown',
    options=[],
    placeholder='Select petition',
    clearable=False,
    style={'width': '100%'}
)

distinct_debate_dates = sorted(upcoming_debate_options['scheduled_debate_date'].unique())

debate_date_dropdown = dcc.Dropdown(
    id='debate-date-dropdown',
    options=[
        {'label': pd.to_datetime(d, dayfirst=True).strftime('%d %b %Y'), 'value': d}
        for d in distinct_debate_dates
    ],
    value=distinct_debate_dates[0] if len(distinct_debate_dates) else None,
    placeholder='Select date' if len(distinct_debate_dates) else 'No scheduled debates',
    disabled=not len(distinct_debate_dates),
    clearable=False,
    style={'width': '210px'}
)

# ── Banner ─────────────────────────────────────────────────

page_nav = dbc.Nav([
    dbc.NavLink("Constituency Overview", id='tab-1-navlink', active=True),
    dbc.NavLink("Petition overview", id='tab-2-navlink', active=False),
    dbc.NavLink("All data", id='tab-3-navlink', active=False),
], pills=True, className="gap-4")

banner = dbc.Navbar(
    dbc.Container([
        html.H3("UK Petitions Dashboard", className="text-white mb-0"),
        page_nav,
        dbc.Row([
            dbc.Col(html.Label("Constituency:", className="text-white mb-0 me-2"), width="auto"),
            dbc.Col(constituency_dropdown, width="auto"),
        ], align="center", className="g-2 flex-nowrap"),
    ], fluid=True, style={'paddingLeft': '34px', 'paddingRight': '32px'}),
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
                                    html.H5(
                                        "Top 5 petitions overall (all constituencies)", className="mb-1 text-center",
                                        style={'minHeight': '40px', 'display': 'flex', 'alignItems': 'flex-start', 'justifyContent': 'center'}
                                    ),
                                    top5_overall_component
                                ], className="pt-2 pb-2"),
                                className="shadow-sm h-100"
                            )
                        ], style={'flex': '0 0 31%', 'maxWidth': '31%'}),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5(
                                        id='top-5-raw-title', className="mb-1 text-center",
                                        style={'minHeight': '40px', 'display': 'flex', 'alignItems': 'flex-start', 'justifyContent': 'center'}
                                    ),
                                    html.Div(id='top-5-table-raw-count')
                                ], className="pt-2 pb-2"),
                                className="shadow-sm h-100"
                            )
                        ], style={'flex': '0 0 31%', 'maxWidth': '31%'}),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5(
                                        id='top5-percent-title', className="mb-1 text-center",
                                        style={'minHeight': '40px', 'display': 'flex', 'alignItems': 'flex-start', 'justifyContent': 'center'}
                                    ),
                                    html.Div(id='top5-percent-table', style={'paddingTop': '2px'})
                                ], className="pt-2 pb-2"),
                                className="shadow-sm h-100"
                            )
                        ], style={'flex': '0 0 38%', 'maxWidth': '38%'})
                    ], className="g-2"),

                    # ── Popular new petitions / Upcoming debates ────────────
                    dbc.Row([
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.Div([
                                        html.H5("Popular new petitions", className="mb-0 me-1"),
                                        html.Span("?", id="popular-petitions-info-icon", style={
                                            'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                                            'width': '16px', 'height': '16px', 'borderRadius': '50%',
                                            'border': '1px solid #6c757d', 'color': '#6c757d',
                                            'fontSize': '11px', 'cursor': 'pointer'
                                        }),
                                        dbc.Tooltip(
                                            "Petitions that have been open for less than a month but already have over 10,000 votes",
                                            target="popular-petitions-info-icon",
                                            placement="top"
                                        )
                                    ], className="mb-3 d-flex align-items-center justify-content-center"),
                                    up_and_coming_component
                                ], className="pt-2 pb-2", style={'paddingLeft': '8px', 'paddingRight': '8px'}),
                                className="shadow-sm h-100"
                            )
                        ], style={'flex': '0 0 39%', 'maxWidth': '39%'}),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.Div([
                                        html.H5("Upcoming debate(s) on", className="mb-0 me-2"),
                                        debate_date_dropdown
                                    ], className="mb-2 d-flex align-items-center justify-content-center"),
                                    html.Div(upcoming_debate_dropdown, className="mb-4"),
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardBody([
                                                    html.H6("No. of sigs in selected constituency", id='debate-constituency-votes-label', className="text-muted",
                                                            style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '12px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                                    html.Div(html.H5(id='debate-constituency-votes-box', className="mb-0"),
                                                             style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                                ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '8px 8px'})
                                            ], className="shadow-sm mb-2", style={'borderRadius': '10px'}),
                                            dbc.Card([
                                                dbc.CardBody([
                                                    html.H6("Ranking based on no. of sigs", className="text-muted",
                                                            style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '12px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                                    html.Div(html.H5(id='debate-ranking-box', className="mb-0"),
                                                             style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                                ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '8px 8px'})
                                            ], className="shadow-sm mb-2", style={'borderRadius': '10px'}),
                                            dbc.Card([
                                                dbc.CardBody([
                                                    html.H6("Signatures per 1,000 population", className="text-muted",
                                                            style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '12px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                                    html.Div(html.H5(id='debate-sig-per-pop-box', className="mb-0"),
                                                             style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                                ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '8px 8px'})
                                            ], className="shadow-sm", style={'borderRadius': '10px'}),
                                        ], style={'flex': '0 0 22%', 'maxWidth': '22%'}),
                                        dbc.Col([
                                            dcc.Graph(
                                                id='upcoming-debates-histogram',
                                                style={'height': '350px'},
                                                config={'displayModeBar': False}
                                            )
                                        ], style={'flex': '0 0 78%', 'maxWidth': '78%'}),
                                    ], className="g-2 mb-2"),
                                ], className="pt-2 pb-2"),
                                className="shadow-sm h-100"
                            )
                        ], style={'flex': '0 0 61%', 'maxWidth': '61%'})
                    ], className="g-2 mt-2"),

                ], style={'padding': '20px'})
            ]),

            dcc.Tab(value='tab-2', children=[
                html.Div([

                    dbc.Row([
                        dbc.Col(
                            html.Label("Select a Petition:", className="fw-bold mb-0"),
                            width="auto", className="d-flex align-items-center"
                        ),
                        dbc.Col(petition_dropdown, width=6),
                    ], className="mb-3 g-2 align-items-center"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Total no. of signatures", className="text-muted",
                                            style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '12px',
                                                   'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                    html.Div(html.H5(id='total-sigs', className="mb-0"),
                                             style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '8px 8px'})
                            ], className="shadow-sm mb-2", style={'borderRadius': '10px'}),
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Scheduled debate date", className="text-muted",
                                            style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '12px',
                                                   'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                    html.Div(html.H5(id='sch-debate-date', className="mb-0"),
                                             style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '8px 8px'})
                            ], className="shadow-sm mb-2", style={'borderRadius': '10px'}),
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Constituency with most signatures", className="text-muted",
                                            style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '12px',
                                                   'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                    html.Div(html.H5(id='highest-count-con', className="mb-0"),
                                             style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '8px 8px'})
                            ], className="shadow-sm", style={'borderRadius': '10px'}),
                        ], width=2),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    dcc.Graph(
                                        id='petition-histogram',
                                        style={'height': '72vh'},
                                        config={'displayModeBar': False}
                                    )
                                ], className="pt-2 pb-2"),
                                className="shadow-sm h-100"
                            )
                        ], width=5),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    dcc.Loading(
                                        id="loading",
                                        type="circle",
                                        children=[mygraph]
                                    )
                                ], className="pt-2 pb-2"),
                                className="shadow-sm h-100"
                            )
                        ], width=5),
                    ], className="g-2")

                ], style={'padding': '20px'})
            ]),

            dcc.Tab(value='tab-3', children=[
                html.Div([

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
                    ])

                ], style={'padding': '20px'})
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
    Output('tab-3-navlink', 'active'),
    Input('tab-1-navlink', 'n_clicks'),
    Input('tab-2-navlink', 'n_clicks'),
    Input('tab-3-navlink', 'n_clicks'),
    prevent_initial_call=True
)
def switch_tab(_n1, _n2, _n3):
    if ctx.triggered_id == 'tab-2-navlink':
        return 'tab-2', False, True, False
    if ctx.triggered_id == 'tab-3-navlink':
        return 'tab-3', False, False, True
    return 'tab-1', True, False, False


# ── Constituency Overview tab ─────────────────────────────

@app.callback(
    Output('top-5-raw-title', 'children'),
    Output('top-5-table-raw-count', 'children'),
    Input('analytics-petition-dropdown', 'value')
)
def update_top5_raw(PCON24CD):
    if PCON24CD is None:
        return "Top 5 petitions in your constituency", html.Div(
            NO_CONSTITUENCY_MESSAGE,
            style={
                'padding': '10px 20px', 'color': '#C0392B', 'fontSize': '16px', 'textAlign': 'center',
                'minHeight': '250px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'
            }
        )

    open_df = petitions_df[
        (petitions_df['PCON24CD'] == PCON24CD) &
        (petitions_df['status'] == 'open')
    ].copy()

    constituency_name = pcon24cds.loc[pcon24cds['PCON24CD'] == PCON24CD, 'constituency_name'].iloc[0]
    title = f"Top 5 petitions in {constituency_name}"

    return title, render_top5_bars(
        open_df, 'signature_count',
        bar_color='#40a583', border_color='#1a7a5c',
        marker_col='median_signature_count'
    )


@app.callback(
    Output('top5-percent-title', 'children'),
    Output('top5-percent-table', 'children'),
    Input('analytics-petition-dropdown', 'value')
)
def update_top5_percent(PCON24CD):
    if PCON24CD is None:
        return "Petitions where your constituency ranks in top 5% of constituencies based on no. of sigs", html.Div(
            NO_CONSTITUENCY_MESSAGE,
            style={
                'padding': '10px 20px', 'color': '#C0392B', 'fontSize': '16px', 'textAlign': 'center',
                'minHeight': '250px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'
            }
        )

    constituency_name = pcon24cds.loc[pcon24cds['PCON24CD'] == PCON24CD, 'constituency_name'].iloc[0]
    title = f"Petitions where {constituency_name} ranks in top 5% of constituencies based on no. of sigs"

    df = petitions_df[
        (petitions_df['PCON24CD'] == PCON24CD) &
        (petitions_df['status'] == 'open') &
        (petitions_df['percentile_rank_raw'] <= 5)
    ][['petition_title', 'petition_url', 'sig_rank_raw', 'percentile_rank_raw',
       'signature_count', 'total_signature_count']] \
        .drop_duplicates(subset='petition_title') \
        .sort_values('percentile_rank_raw', ascending=True) \
        .copy()

    if df.empty:
        return title, html.Div(
            "No petitions currently have this constituency in the top 5% of signatures.",
            style={'padding': '10px 20px', 'color': '#777', 'fontSize': '13px'}
        )

    df['petition_title_link'] = df.apply(
        lambda r: f"[{r['petition_title']}]({r['petition_url']})", axis=1
    )
    df['sig_rank_raw'] = df['sig_rank_raw'].astype(int)
    df['sig_ratio_display'] = df.apply(
        lambda r: f"{r['signature_count']:,} of  \n{r['total_signature_count']:,} sigs", axis=1
    )
    df['rank_display'] = df.apply(
        lambda r: f"{r['sig_rank_raw']} of {TOTAL_CONSTITUENCIES}", axis=1
    )

    table = dag.AgGrid(
        id='top5-percent-datatable',
        rowData=df.to_dict('records'),
        columnDefs=[
            {'field': 'petition_title_link', 'headerName': 'Petition', 'cellRenderer': 'markdown',
             'cellClass': 'petition-title-cell',
             'flex': 2, 'minWidth': 220, 'wrapText': True, 'autoHeight': True},
            {'field': 'rank_display', 'headerName': 'Ranking based on no. of sigs', 'cellRenderer': 'markdown',
             'cellClass': 'rank-ratio-cell', 'headerClass': 'ag-header-center',
             'flex': 1, 'minWidth': 110, 'wrapText': True, 'autoHeight': True},
            {'field': 'sig_ratio_display', 'headerName': 'No. of sigs in constituency', 'cellRenderer': 'markdown',
             'cellClass': 'sig-ratio-cell', 'headerClass': 'ag-header-center',
             'flex': 1.4, 'minWidth': 150, 'wrapText': True, 'autoHeight': True},
        ],
        defaultColDef={'sortable': False, 'resizable': False, 'wrapHeaderText': True, 'autoHeaderHeight': True},
        dashGridOptions={
            'domLayout': 'autoHeight',
            'onRowDataUpdated': {'function': 'setTimeout(function(){ params.api.resetRowHeights(); }, 50)'},
            'onFirstDataRendered': {'function': 'setTimeout(function(){ params.api.resetRowHeights(); }, 50)'},
        },
        dangerously_allow_code=True,
        className='ag-theme-alpine',
        style={'width': '100%'},
    )

    # Grid sizes itself to however many rows there are, but never grows past the
    # height the card was originally designed for — it scrolls internally instead.
    return title, html.Div(table, style={'maxHeight': '435px', 'overflowY': 'auto'})


@app.callback(
    Output('upcoming-debate-dropdown', 'options'),
    Output('upcoming-debate-dropdown', 'value'),
    Output('upcoming-debate-dropdown', 'placeholder'),
    Output('upcoming-debate-dropdown', 'disabled'),
    Input('debate-date-dropdown', 'value')
)
def update_petitions_for_date(selected_date):
    if selected_date is None:
        return [], None, 'No scheduled debates', True

    matches = upcoming_debate_options[upcoming_debate_options['scheduled_debate_date'] == selected_date]
    options = [
        {'label': row['petition_title'], 'value': row['petition_id']}
        for _, row in matches.iterrows()
    ]
    value = matches.iloc[0]['petition_id'] if len(matches) else None
    return options, value, 'Select petition', False


@app.callback(
    Output('debate-constituency-votes-box', 'children'),
    Output('debate-constituency-votes-label', 'children'),
    Output('debate-sig-per-pop-box', 'children'),
    Output('debate-ranking-box', 'children'),
    Output('upcoming-debates-histogram', 'figure'),
    Input('upcoming-debate-dropdown', 'value'),
    Input('analytics-petition-dropdown', 'value')
)
def update_debate_section(petition_id, PCON24CD):
    no_constituency = html.Span("Select a constituency", style={'color': '#C0392B'})
    if petition_id is None:
        return "", "No. of sigs in selected constituency", "", "", go.Figure()

    df = petitions_df[petitions_df['petition_id'] == petition_id]

    total_votes = df['total_signature_count'].iloc[0]

    constituency_row = df.loc[df['PCON24CD'] == PCON24CD] if PCON24CD is not None else None

    constituency_votes = constituency_row['signature_count'].iloc[0] if constituency_row is not None else None
    sig_per_pop = constituency_row['sig_per_pop'].iloc[0] if constituency_row is not None else None
    sig_rank = constituency_row['sig_rank_raw'].iloc[0] if constituency_row is not None else None
    sig_percentile = constituency_row['percentile_rank_raw'].iloc[0] if constituency_row is not None else None

    constituency_label = (
        f"No. of sigs in {pcon24cds.loc[pcon24cds['PCON24CD'] == PCON24CD, 'constituency_name'].iloc[0]}"
        if PCON24CD is not None else "No. of sigs in selected constituency"
    )

    fig = render_signature_histogram(df, df['median_signature_count'].iloc[0], constituency_votes)

    paren_style = {'display': 'block', 'marginTop': '6px', 'fontSize': '14px', 'fontWeight': 'normal', 'color': '#888'}

    constituency_votes_str = (
        html.Span([
            f"{constituency_votes:,} of", html.Br(), f"{total_votes:,} sigs",
            html.Span(f"({(constituency_votes / total_votes) * 100:.2f}%)", style=paren_style)
        ])
        if constituency_votes is not None else no_constituency
    )
    sig_per_pop_str = f"{sig_per_pop:.2f}" if sig_per_pop is not None else no_constituency
    category = percentile_category(sig_percentile)
    ranking_str = (
        html.Span([f"{int(sig_rank)} of {TOTAL_CONSTITUENCIES}"] + ([html.Span(f"({category})", style=paren_style)] if category else []))
        if sig_rank is not None else no_constituency
    )

    return constituency_votes_str, constituency_label, sig_per_pop_str, ranking_str, fig


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
    df['opened_at'] = pd.to_datetime(df['opened_at'], dayfirst=True).dt.date
    df['months_open'] = df['opened_at'].apply(lambda d: (today - d).days).apply(
        lambda n: 'Less than 1 month' if n < 30 else '1-3 months' if n < 90 else '4-6 months' if n < 180 else '6+ months'
    )

    # Debate date formatting
    df['scheduled_debate_date'] = pd.to_datetime(df['scheduled_debate_date'], dayfirst=True, errors='coerce').dt.date

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
                    {'field': 'percentile_rank_raw', 'headerName': 'Percentile ranking (counts)', 'flex': 1, 'minWidth': 130,
                     'valueFormatter': PERCENTILE_CATEGORY_FORMAT},
                ]
            },
            {
                'headerName': 'Counts per 1,000 population',
                'children': [
                    {'field': 'sig_per_pop', 'headerName': 'Signatures per 1,000', 'flex': 1, 'minWidth': 130},
                    {'field': 'percentile_rank_pop', 'headerName': 'Percentile ranking (sig per 1000)', 'flex': 1, 'minWidth': 130,
                     'valueFormatter': PERCENTILE_CATEGORY_FORMAT},
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
    Output('petition-histogram', 'figure'),
    Output('total-sigs', 'children'),
    Output('sch-debate-date', 'children'),
    Output('highest-count-con', 'children'),
    Input('petition-dropdown', 'value'),
    Input('analytics-petition-dropdown', 'value')
)
def update_graph(petition_id, PCON24CD):
    callback_start = time.time()

    cached_data = get_petition_data(petition_id)

    df = pd.DataFrame(cached_data, columns=petitions_df.columns)
    print(f"Time to retrieve and process petition data: {time.time() - callback_start:.4f}s")

    total_signatures = df['signature_count'].sum()

    sch_debate_date = df['scheduled_debate_date'].iloc[0] if 'scheduled_debate_date' in df.columns else None
    debate_date_str = str(sch_debate_date) if pd.notna(sch_debate_date) else "Not scheduled"

    max_row = df.loc[df['signature_count'].idxmax()]
    highest_count_con = max_row['constituency_name']
    highest_count = max_row['signature_count']

    constituency_row = df.loc[df['PCON24CD'] == PCON24CD] if PCON24CD is not None else None
    constituency_votes = constituency_row['signature_count'].iloc[0] if constituency_row is not None else None
    histogram_fig = render_signature_histogram(df, df['median_signature_count'].iloc[0], constituency_votes)

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
        projection_scale=0.9,
        center=dict(lat=54.5, lon=-3),
        lataxis_range=[48, 60],
        lonaxis_range=[-10, 4]
    )

    # The colour scale is capped at the 95th percentile (max_color) rather than the true
    # max so a single outlier constituency doesn't wash out the rest of the map, but the
    # top tick should still read as the true highest count rather than that cap value.
    # Intermediate ticks are only kept when they land exactly on a whole number, rather
    # than being rounded (which could otherwise show duplicate labels).
    raw_tickvals = np.linspace(0, max_color, 5)
    is_whole = np.isclose(raw_tickvals, np.round(raw_tickvals))
    is_whole[-1] = True  # always keep the top tick so it can show the true max

    colorbar_tickvals = raw_tickvals[is_whole]
    colorbar_ticktext = [f"{int(round(v)):,}" for v in colorbar_tickvals]
    colorbar_ticktext[-1] = f"{highest_count:,}"

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 20},
        uirevision='constant',
        coloraxis_colorbar=dict(
            title="No. of sigs",
            orientation="h",
            thickness=15,
            len=0.7,
            x=0.5,
            xanchor="center",
            y=0,
            yanchor="top",
            tickmode="array",
            tickvals=colorbar_tickvals,
            ticktext=colorbar_ticktext
        )
    )

    print(f"Time to plot the choropleth: {time.time() - plot_start:.4f}s")

    return (
        fig,
        histogram_fig,
        f"{total_signatures:,}",
        debate_date_str,
        f"{highest_count_con} ({highest_count:,})"
    )


if __name__ == '__main__':
    app.run(debug=(ENV == 'local'), port=8051)