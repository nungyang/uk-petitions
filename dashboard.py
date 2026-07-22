####################
#### Setting up ####
####################

# ── Imports ───────────────────────────────────────────────

import time
_startup_t0 = time.time()

import os
import gc
import textwrap
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache

import boto3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, Output, Input, html, ctx
from flask import Response
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

print(f"[startup] Imports done: {time.time() - _startup_t0:.2f}s")


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


# ── Cached data loaders ───────────────────────────────────

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
    # England & Wales + Northern Ireland + Scotland population estimates, combined
    # into one lookup. Scotland's file comes from a different source (NRS rather
    # than ONS) but uses the same PCON24CD/pop column names as the other two.
    if ENV == 'local':
        print("Loading population data from local cache...")
        main_df = pd.read_csv(script_dir / 'cached_data' / 'pop_estimate_2021.csv')
        ni_df = pd.read_csv(script_dir / 'cached_data' / 'pop_estimate_2021_ni.csv')
        sct_df = pd.read_csv(script_dir / 'cached_data' / 'pop_estimate_2021_sct.csv')
    else:
        main_df = load_csv('static data/pop_estimate_2021.csv')
        ni_df = load_csv('static data/pop_estimate_2021_ni.csv')
        sct_df = load_csv('static data/pop_estimate_2021_sct.csv')
    # Column order differs between the files; concat aligns by column name, not
    # position, so that's fine as long as the names match (they do).
    return pd.concat([main_df, ni_df, sct_df], ignore_index=True)


@lru_cache(maxsize=128)
def get_petition_data(petition_id):
    start = time.time()
    df = petitions_df[petitions_df['petition_id'] == petition_id]
    print(f"Time to retrieve petition data from cache: {time.time() - start:.4f}s")
    return tuple(df.itertuples(index=False))


# ── Data processing ───────────────────────────────────────

print(f"Environment: {ENV}")

_data_load_t0 = time.time()
print("Loading petitions data...")
petitions_list, petitions_count = get_petitions_data()

print("Loading population data...")
pop_df = get_population_data()

print("Done loading data.")
print(f"[startup] S3/local data download: {time.time() - _data_load_t0:.2f}s")

_merge_t0 = time.time()
# Data pull has missing rows if value should actually be 0 so making sure they get filled with 0
petition_ids = petitions_list[['petition_id']].drop_duplicates()
pcon24cds = petitions_count[['PCON24CD', 'constituency_name']].drop_duplicates()
TOTAL_CONSTITUENCIES = len(pcon24cds)
# Scotland's population estimates come from a different source (NRS) than the
# rest of the UK (ONS) — see get_population_data — so its constituencies are
# ranked in their own pool for sig_per_pop_rank rather than pooled with the rest.
_is_scotland_pop = pop_df['PCON24CD'].str.startswith('S')
TOTAL_CONSTITUENCIES_SCOTLAND = pop_df.loc[_is_scotland_pop, 'PCON24CD'].nunique()
TOTAL_CONSTITUENCIES_WITH_POP = pop_df.loc[~_is_scotland_pop, 'PCON24CD'].nunique()

skeleton_df = petition_ids.merge(pcon24cds, how='cross')

petitions_count = skeleton_df.merge(petitions_count.drop(columns=['constituency_name']), on = ['petition_id', 'PCON24CD'], how = 'left')
petitions_count['signature_count'] = petitions_count['signature_count'].fillna(0).astype(int)

# output_path = script_dir / "cached_data" / "file_to_check.csv"
# petitions_count.to_csv(output_path, index=False)

# Adding data on population
petitions_count = petitions_count.merge(pop_df[['PCON24CD', 'pop']], on= ['PCON24CD'], how='left')

# Adding column on sig per pop
petitions_count['sig_per_pop'] = (petitions_count['signature_count'] / petitions_count['pop']) * 1000
# 'pop' itself is only ever used to derive sig_per_pop above — nothing else reads it,
# so there's no reason for it to keep taking up space in petitions_df from here on.
petitions_count = petitions_count.drop(columns=['pop'])

# Adding rank — Scotland is ranked in its own pool, separate from the rest of the
# UK (see TOTAL_CONSTITUENCIES_SCOTLAND above), for both sig_per_pop_rank and its
# percentile.
petitions_count['is_scotland'] = petitions_count['PCON24CD'].str.startswith('S')
petitions_count['sig_per_pop_rank'] = petitions_count.groupby(['petition_id', 'is_scotland'])['sig_per_pop'].rank(ascending=False, method='min')
petitions_count['sig_rank_raw'] = petitions_count.groupby('petition_id')['signature_count'].rank(ascending=False, method='min')
petitions_count['percentile_rank_raw'] = (100 - (petitions_count.groupby('petition_id')['signature_count'].rank(pct=True) * 100)).round(1)
petitions_count['percentile_rank_pop'] = (100 - (petitions_count.groupby(['petition_id', 'is_scotland'])['sig_per_pop'].rank(pct=True) * 100)).round(1)
# 'is_scotland' itself is only ever used to scope the two rankings above — nothing
# else reads it, so it doesn't need to keep taking up space in petitions_df.
petitions_count = petitions_count.drop(columns=['is_scotland'])

# Working out median count for each petition
median_counts = petitions_count.groupby('petition_id')['signature_count'].median().reset_index()
median_counts.columns = ['petition_id', 'median_signature_count']

# Adding median counts to petitions list
petitions_list = petitions_list.merge(median_counts, on='petition_id', how='left')

# Merging petitions count to petitions list
petitions_df = petitions_list.merge(petitions_count, on='petition_id', how='left')

# petitions_count/skeleton_df/pop_df/petition_ids/median_counts were only ever
# stepping stones toward petitions_df — everything they held is now merged in,
# and nothing downstream reads them again, so free the memory rather than
# leaving them resident as unused module-level globals for the app's lifetime.
del petitions_count, skeleton_df, pop_df, petition_ids, median_counts
gc.collect()

# If total count is less than 10,000, then remove ranking
petitions_df.loc[petitions_df['total_signature_count'] <= 10000, 'percentile_rank_raw'] = np.nan
petitions_df.loc[petitions_df['total_signature_count'] <= 10000, 'percentile_rank_pop'] = np.nan
petitions_df.loc[petitions_df['total_signature_count'] <= 10000, 'sig_rank_raw'] = np.nan
petitions_df.loc[petitions_df['total_signature_count'] <= 10000, 'sig_per_pop_rank'] = np.nan

print(f"[startup] Merge/cross-join/rank computation: {time.time() - _merge_t0:.2f}s")

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

# Displays a rank (e.g. sig_rank_raw / sig_per_pop_rank) as "N of {TOTAL_CONSTITUENCIES}"
# while the underlying cell value stays numeric, so ascending/descending sort works.
RANK_DISPLAY_FORMAT = {'function': (
    f"params.value == null || params.value === 0 ? '' : params.value + ' of {TOTAL_CONSTITUENCIES} constituencies'"
)}

def _is_scotland(PCON24CD):
    return PCON24CD is not None and PCON24CD.startswith('S')


# Same as RANK_DISPLAY_FORMAT, but for sig_per_pop_rank specifically — Scotland is
# ranked in its own pool (see TOTAL_CONSTITUENCIES_SCOTLAND), so the denominator
# depends on whether the constituency being displayed is Scottish.
def make_rank_display_format_pop(PCON24CD):
    total = TOTAL_CONSTITUENCIES_SCOTLAND if _is_scotland(PCON24CD) else TOTAL_CONSTITUENCIES_WITH_POP
    return {'function': (
        f"params.value == null || params.value === 0 ? '' : params.value + ' of {total} constituencies'"
    )}

# "?" info icons appended to the "Ranking based on no. of sigs" / ".../sigs/pop"
# headers, explaining via a dbc.Tooltip (added alongside the table below) why a
# petition's ranking can be blank — it's suppressed for petitions with under 10,000
# signatures (see the total_signature_count <= 10000 filtering above). A dbc.Tooltip
# is used instead of a native title attribute because native tooltips are unreliable —
# some browsers/embedded webviews cancel them if the hovered element's ancestors mutate
# (as AG Grid's header cells do while re-laying-out after a constituency is selected).
# The icon itself still needs AG Grid's header template override (the default
# agColumnHeader template, docs' "Header Templates" page, with one extra <span> spliced
# in) since headerName is plain text and can't hold markup.
RANK_INFO_TEXT = "Ranking only shows for petitions with 10,000 or more signatures"
SIG_RANK_RAW_INFO_ICON_ID = 'sig-rank-raw-info-icon'
SIG_PER_POP_RANK_INFO_ICON_ID = 'sig-per-pop-rank-info-icon'
# The sig/pop ranking specifically also ranks Scotland separately from the rest
# of the UK (its population estimates come from a different source — see
# get_population_data), on top of the 10,000-signature rule both rank columns
# share. Two <br>s (not one) so it reads as a separate paragraph.
SIG_PER_POP_RANK_INFO_TEXT = [
    RANK_INFO_TEXT,
    html.Br(), html.Br(),
    f"Constituencies in Scotland are ranked separately, out of the other {TOTAL_CONSTITUENCIES_SCOTLAND} "
    "Scottish constituencies, because their population estimates come from a different source.",
]


def make_header_info_icon_template(icon_id):
    return f'''
<div class="ag-cell-label-container" role="presentation">
  <span data-ref="eMenu" class="ag-header-icon ag-header-cell-menu-button"></span>
  <span data-ref="eFilterButton" class="ag-header-icon ag-header-cell-filter-button"></span>
  <div data-ref="eLabel" class="ag-header-cell-label" role="presentation">
    <span class="header-title-wrap"><span data-ref="eText" class="ag-header-cell-text"></span><span class="header-info-icon" id="{icon_id}">?</span></span>
    <span data-ref="eFilter" class="ag-header-icon ag-filter-icon"></span>
    <span data-ref="eSortOrder" class="ag-header-icon ag-sort-order"></span>
    <span data-ref="eSortAsc" class="ag-header-icon ag-sort-ascending-icon"></span>
    <span data-ref="eSortDesc" class="ag-header-icon ag-sort-descending-icon"></span>
    <span data-ref="eSortNone" class="ag-header-icon ag-sort-none-icon"></span>
  </div>
</div>
'''


SIG_RANK_RAW_HEADER_TEMPLATE = make_header_info_icon_template(SIG_RANK_RAW_INFO_ICON_ID)
SIG_PER_POP_RANK_HEADER_TEMPLATE = make_header_info_icon_template(SIG_PER_POP_RANK_INFO_ICON_ID)

# Same pattern for the "Scheduled debate date" header — explains the 100,000-signature
# debate threshold and the greyed-out-past-debate styling (see .past-debate-date).
# Two <br>s (not one) between the sentences so they read as separate paragraphs
# rather than two lines of the same one.
SCHEDULED_DEBATE_INFO_TEXT = [
    "Petitions with 100,000 signatures or more are considered for debate by the Petitions Committee.",
    html.Br(), html.Br(),
    "Debates that have already taken place are greyed out",
]
SCHEDULED_DEBATE_INFO_ICON_ID = 'scheduled-debate-info-icon'
SCHEDULED_DEBATE_HEADER_TEMPLATE = make_header_info_icon_template(SCHEDULED_DEBATE_INFO_ICON_ID)

# Same pattern for the "Months open" header.
MONTHS_OPEN_INFO_TEXT = "Petitions are open for 6 months"
MONTHS_OPEN_INFO_ICON_ID = 'months-open-info-icon'
MONTHS_OPEN_HEADER_TEMPLATE = make_header_info_icon_template(MONTHS_OPEN_INFO_ICON_ID)

# Displays a months_open_rank (0-3) as its label, while the underlying cell value
# stays numeric, so ascending/descending sort follows chronological order.
MONTHS_OPEN_FORMAT = {'function': (
    "params.value === 0 ? '< 1 month' : "
    "params.value === 1 ? '1-3 months' : "
    "params.value === 2 ? '4-6 months' : '6+ months'"
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


# ── All Petitions table (shared builders) ──────────────────
#
# These build the "All data" ag-Grid's rowData/columnDefs. They're shared between
# the initial page layout (built once, at import time) and the constituency-change
# callback, which patches the already-mounted grid's rowData/columnDefs directly
# instead of replacing the whole component — recreating the grid from scratch on
# every constituency click was the main cause of the slow reloads, since it forced
# ag-Grid to fully unmount/remount (recompiling every dangerously_allow_code
# function string) and re-transfer all ~2,300 rows even though most columns don't
# change between one constituency and the next.

@lru_cache(maxsize=4)
def _petitions_display_base(today):
    """Petition-level columns that don't depend on the selected constituency
    (title link, dates, months-open, debate info). Cached per calendar day so
    repeated constituency clicks on the same day skip these pandas .apply() calls."""
    df = petitions_list.copy()
    df['petition_id'] = df['petition_id'].astype(str)

    # Months open, based on actual calendar months from the open date (e.g. opened
    # 5 July: "< 1 month" until 5 August, "1-3 months" until 5 October,
    # "4-6 months" up to and including 5 January, then "6+ months"). Stored as a
    # numeric rank (0-3) rather than the display string so the grid's default sort
    # follows chronological order instead of alphabetical; MONTHS_OPEN_FORMAT maps
    # the rank back to its label.
    df['opened_at'] = pd.to_datetime(df['opened_at']).dt.date
    df['months_open_rank'] = df['opened_at'].apply(
        lambda d: 0 if today < d + relativedelta(months=1)
        else 1 if today < d + relativedelta(months=3)
        else 2 if today <= d + relativedelta(months=6)
        else 3
    )

    # Debate date formatting. Petitions with 100,000+ signatures are considered
    # for a Commons debate even before one is actually scheduled.
    df['scheduled_debate_date'] = pd.to_datetime(df['scheduled_debate_date'], errors='coerce').dt.date
    df['is_past_debate'] = df['scheduled_debate_date'].notna() & (df['scheduled_debate_date'] < today)
    df['debate_display'] = df.apply(
        lambda r: r['scheduled_debate_date'] if pd.notna(r['scheduled_debate_date'])
        else ('To be considered for debate' if r['total_signature_count'] >= 100000 else 'N/A'),
        axis=1
    )
    # Numeric sort key so actual dates always sort before "To be considered for
    # debate" (and that before blanks) in both directions — a real comparator
    # isn't usable here (dash_ag_grid only supports single-argument value-style
    # functions, not AG Grid's multi-arg comparator signature).
    df['debate_sort_key'] = df.apply(
        lambda r: r['scheduled_debate_date'].toordinal() if pd.notna(r['scheduled_debate_date'])
        else (-1 if r['total_signature_count'] >= 100000 else -2),
        axis=1
    )

    # Clickable title as markdown
    df['petition_title_link'] = df.apply(
        lambda r: f"[{r['petition_title']}]({r['petition_url']})", axis=1
    )
    return df


def _build_all_petitions_rowdata(PCON24CD):
    today = datetime.now().date()

    # Constituency-level stats for the selected constituency
    constituency_sigs = petitions_df[petitions_df['PCON24CD'] == PCON24CD][[
        'petition_id', 'signature_count', 'sig_rank_raw',
        'percentile_rank_raw', 'percentile_rank_pop',
        'sig_per_pop', 'sig_per_pop_rank'
    ]].drop_duplicates(subset='petition_id').copy()
    constituency_sigs['petition_id'] = constituency_sigs['petition_id'].astype(str)

    df = _petitions_display_base(today).merge(constituency_sigs, on='petition_id', how='left')

    if PCON24CD is not None:
        df['signature_count'] = df['signature_count'].astype(int)
        # sig_rank_raw/sig_per_pop_rank are NaN for petitions with <= 10,000 total
        # signatures (rank suppressed above). astype(int) can't hold NaN, so cast
        # element-wise and leave those as None instead of crashing.
        df['sig_rank_raw'] = df['sig_rank_raw'].apply(lambda x: int(x) if pd.notna(x) else None)
        df['sig_per_pop_rank'] = df['sig_per_pop_rank'].apply(lambda x: int(x) if pd.notna(x) else None)
    df['percentile_rank_raw'] = df['percentile_rank_raw'].round(1)
    df['percentile_rank_pop'] = df['percentile_rank_pop'].round(1)
    df['sig_per_pop'] = df['sig_per_pop'].round(2)
    df['sig_prop_of_total'] = df['signature_count'] / df['total_signature_count'] * 100

    table_df = df[[
        'petition_title_link',
        'opened_at',
        'months_open_rank',
        'total_signature_count',
        'signature_count',
        'sig_rank_raw',
        'percentile_rank_raw',
        'sig_prop_of_total',
        'sig_per_pop',
        'sig_per_pop_rank',
        'percentile_rank_pop',
        'debate_display',
        'debate_sort_key',
        'is_past_debate'
    ]].sort_values('total_signature_count', ascending=False)

    return table_df.to_dict('records')


_ALL_PETITIONS_PAGE_SIZE = 20

# When no constituency is selected, a row near the top of each page carries the
# merged placeholder message (colSpan across all 7 per-constituency columns);
# every other row's cell in that column renders blank. dash_ag_grid compiles
# colSpan/valueGetter as an expression-bodied arrow function (params) => (CODE) —
# no statements/var/return allowed — so this has to be one composed expression.
_ALL_PETITIONS_NEAR_TOP_OFFSET = 1
_all_petitions_page_start = f"(Math.floor(params.node.rowIndex / {_ALL_PETITIONS_PAGE_SIZE}) * {_ALL_PETITIONS_PAGE_SIZE})"
_all_petitions_target_row = f"({_all_petitions_page_start} + {_ALL_PETITIONS_NEAR_TOP_OFFSET})"
_ALL_PETITIONS_IS_TARGET_ROW = f"(params.node.rowIndex === {_all_petitions_target_row})"

_ALL_PETITIONS_NUMBER_FORMAT = {'function': "d3.format(',')(params.value)"}


def _build_all_petitions_columndefs(PCON24CD):
    # When no constituency is selected, the seven per-constituency columns lose their
    # internal grid lines (both the vertical divider between them and the row line
    # under them) so they read as one blank panel instead of seven empty columns;
    # once a constituency is picked they get their normal gridlines back.
    SPAN_GROUP_CELL_CLASS = 'span-group-cell' if PCON24CD is None else ''
    SPAN_GROUP_HEADER_CLASS = 'span-group-header' if PCON24CD is None else ''

    signature_count_coldef = (
        {'field': 'signature_count', 'headerName': 'No. of sigs in constituency',
         'cellRenderer': 'markdown', 'cellClass': f'no-constituency-message {SPAN_GROUP_CELL_CLASS}',
         'headerClass': SPAN_GROUP_HEADER_CLASS,
         'valueGetter': {'function': f"{_ALL_PETITIONS_IS_TARGET_ROW} ? 'Select a constituency  \\n(see top right)' : ''"},
         'colSpan': {'function': f"{_ALL_PETITIONS_IS_TARGET_ROW} ? 7 : 1"},
         'flex': 1, 'minWidth': 160}
        if PCON24CD is None else
        {'field': 'signature_count', 'headerName': 'No. of sigs in constituency',
         'valueFormatter': _ALL_PETITIONS_NUMBER_FORMAT, 'flex': 1, 'minWidth': 160}
    )

    return [
        {'field': 'petition_title_link', 'headerName': 'Petition', 'cellRenderer': 'markdown',
         'filter': 'agTextColumnFilter',
         'filterParams': {'filterOptions': ['contains', 'notContains']},
         'cellClass': 'petition-title-cell',
         'sortable': False,
         'cellStyle': {'textAlign': 'left'},
         'flex': 1.6, 'minWidth': 220, 'wrapText': True, 'autoHeight': True},
        {'field': 'opened_at', 'headerName': 'Date opened', 'flex': 0.9, 'minWidth': 125},
        {'field': 'months_open_rank', 'headerName': 'Months open', 'flex': 0.8, 'minWidth': 110,
         'valueFormatter': MONTHS_OPEN_FORMAT,
         'headerComponentParams': {'template': MONTHS_OPEN_HEADER_TEMPLATE}},
        {'field': 'total_signature_count', 'headerName': 'Total no. of sigs',
            'valueFormatter': _ALL_PETITIONS_NUMBER_FORMAT, 'flex': 1, 'minWidth': 130},
        {'field': 'debate_sort_key', 'headerName': 'Scheduled debate\ndate', 'flex': 0.65, 'minWidth': 140,
         'valueFormatter': {'function': "params.data.debate_display || ''"},
         'cellClass': {'function': "'debate-cell' + (params.data.is_past_debate ? ' past-debate-date' : '')"},
         'wrapText': True, 'autoHeight': True,
         'headerComponentParams': {'template': SCHEDULED_DEBATE_HEADER_TEMPLATE}},
        {
            'headerName': 'Metrics based on no. of sigs',
            'headerClass': 'centered-group-header',
            'children': [
                signature_count_coldef,
                {'field': 'sig_rank_raw', 'headerName': 'Ranking based on no. of sigs',
                 'valueFormatter': RANK_DISPLAY_FORMAT, 'cellClass': SPAN_GROUP_CELL_CLASS,
                 'headerClass': SPAN_GROUP_HEADER_CLASS, 'flex': 1, 'minWidth': 170,
                 'wrapText': True, 'autoHeight': True,
                 'headerComponentParams': {'template': SIG_RANK_RAW_HEADER_TEMPLATE}},
                {'field': 'percentile_rank_raw', 'headerName': 'Ranking as percentile', 'flex': 1, 'minWidth': 150,
                 'valueFormatter': PERCENTILE_CATEGORY_FORMAT, 'cellClass': SPAN_GROUP_CELL_CLASS,
                 'headerClass': SPAN_GROUP_HEADER_CLASS},
                {'field': 'sig_prop_of_total', 'headerName': 'No. of sigs as prop of all sigs (%)', 'flex': 1, 'minWidth': 170,
                 'valueFormatter': {'function': "params.value == null ? '' : params.value.toFixed(2) + '%'"},
                 'cellClass': SPAN_GROUP_CELL_CLASS, 'headerClass': SPAN_GROUP_HEADER_CLASS},
            ]
        },
        {
            'headerName': 'Metrics based on avg no. of sigs',
            'headerClass': 'centered-group-header',
            'children': [
                {'field': 'sig_per_pop', 'headerName': 'Avg no. of sigs per 1000 pop', 'flex': 1, 'minWidth': 150,
                 'cellClass': SPAN_GROUP_CELL_CLASS, 'headerClass': SPAN_GROUP_HEADER_CLASS},
                {'field': 'sig_per_pop_rank', 'headerName': 'Ranking based on no. of sigs/pop',
                 'valueFormatter': make_rank_display_format_pop(PCON24CD), 'cellClass': SPAN_GROUP_CELL_CLASS,
                 'headerClass': SPAN_GROUP_HEADER_CLASS, 'flex': 1, 'minWidth': 170,
                 'wrapText': True, 'autoHeight': True,
                 'headerComponentParams': {'template': SIG_PER_POP_RANK_HEADER_TEMPLATE}},
                {'field': 'percentile_rank_pop', 'headerName': 'Ranking as percentile', 'flex': 1, 'minWidth': 170,
                 'cellClass': SPAN_GROUP_CELL_CLASS, 'headerClass': SPAN_GROUP_HEADER_CLASS,
                 'valueFormatter': PERCENTILE_CATEGORY_FORMAT},
            ]
        },
    ]


def build_all_petitions_table():
    """Builds the "All data" tab's ag-Grid once, for the initial (no constituency
    selected) state. Later constituency changes patch this same mounted grid's
    rowData/columnDefs via a callback rather than replacing it."""
    table = dag.AgGrid(
        id='all-petitions-datatable',
        rowData=_build_all_petitions_rowdata(None),
        columnDefs=_build_all_petitions_columndefs(None),
        defaultColDef={'sortable': True, 'resizable': False, 'wrapHeaderText': True, 'autoHeaderHeight': True,
                       'cellStyle': {'textAlign': 'center'}},
        dashGridOptions={'pagination': True, 'paginationPageSize': _ALL_PETITIONS_PAGE_SIZE, 'domLayout': 'autoHeight',
                         'unSortIcon': True, 'groupHeaderHeight': 56, 'enableCellSpan': True},
        dangerously_allow_code=True,
        className='ag-theme-alpine',
        style={'width': '100%'},
    )

    return html.Div([
        table,
        dbc.Tooltip(RANK_INFO_TEXT, target=SIG_RANK_RAW_INFO_ICON_ID, placement='top'),
        dbc.Tooltip(SIG_PER_POP_RANK_INFO_TEXT, target=SIG_PER_POP_RANK_INFO_ICON_ID, placement='top'),
        dbc.Tooltip(SCHEDULED_DEBATE_INFO_TEXT, target=SCHEDULED_DEBATE_INFO_ICON_ID, placement='top'),
        dbc.Tooltip(MONTHS_OPEN_INFO_TEXT, target=MONTHS_OPEN_INFO_ICON_ID, placement='top'),
    ], style={'overflowX': 'auto', 'width': '100%'})


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


def render_signature_histogram(df, median_value, highlight_value=None, constituency_name=None):
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
        hoverlabel=dict(bgcolor='white', font_color='#333'),
        showlegend=False
    ))

    # Legend-only entries: these traces plot nothing (x/y are None) but still add a
    # swatch to the legend, since the bar trace above mixes colours across bins and
    # can't carry its own legend entry. Added in display order (Median first).
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines', line=dict(width=2, color='#D55E00'),
        name='Median', showlegend=True, hoverinfo='skip'
    ))
    highlight_label = f"Interval that includes {constituency_name}" if constituency_name else "Interval that includes selected constituency"
    # Wrap long labels (e.g. long constituency names) onto multiple lines rather than
    # letting the legend box grow wider — Plotly grows a multi-line entry's row height
    # automatically, so this caps the legend's width and lets it grow taller instead.
    highlight_label = '<br>'.join(textwrap.wrap(highlight_label, width=28))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers', marker=dict(size=10, symbol='square', color='#40a583'),
        name=highlight_label, showlegend=True, hoverinfo='skip'
    ))

    fig.update_layout(
        hovermode='x',
        dragmode=False,
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
        legend=dict(
            x=1, y=1, xanchor='right', yanchor='top',
            bgcolor='rgba(255,255,255,0.85)', bordercolor='#dee2e6', borderwidth=1,
            font=dict(size=11)
        ),
    )
    fig.add_vline(x=median_value, line_width=2, line_color='#D55E00')

    return fig


####################
#### App setup  ####
####################

# ── Initialise app ────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.PULSE],
    suppress_callback_exceptions=True,
    title="Overview of UK Petitions | UK Parliament Petitions Dashboard",
    meta_tags=[
        {"name": "google-site-verification", "content": "CQ5IOretsHhN61VXff8LDPQGm1v66PA8JXIAPbtgdy4"},
        {
            "name": "description",
            "content": "Constituency level overview of UK Parliament petitions, including most popular petitions, number of signatures and ranking of petitions",
        },
        {"property": "og:title", "content": "Overview of UK Petitions | UK Parliament Petitions Dashboard"},
        {
            "property": "og:description",
            "content": "Constituency level overview of UK Parliament petitions, including most popular petitions, number of signatures and ranking of petitions",
        },
        {"property": "og:type", "content": "website"},
        {"property": "og:url", "content": "https://uk-petitions-dashboard.up.railway.app/"},
        {
            "property": "og:image",
            "content": "https://uk-petitions-dashboard.up.railway.app/assets/dashboard_screenshot.png",
        },
    ],
)
server = app.server


@server.route("/robots.txt")
def robots_txt():
    return Response("User-agent: *\nAllow: /\n", mimetype="text/plain")


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
            #top5-percent-datatable .ag-center-cols-container,
            #all-petitions-datatable .ag-center-cols-viewport,
            #all-petitions-datatable .ag-center-cols-container {
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
            #upcoming-debate-dropdown .Select.is-open.has-value .Select-value,
            #petition-dropdown .Select.is-open.has-value .Select-value {
                display: none;
            }

            /* dbc.NavLink has no href (tab switching is driven by a callback off its
               id, not real navigation), so browsers don't apply the usual link
               pointer cursor and fall back to the text-select cursor instead. */
            #tab-1-navlink, #tab-2-navlink, #tab-3-navlink {
                cursor: pointer;
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

            /* Wrap ag-Grid header text only at word boundaries, never mid-word or ellipsis.
               pre-line (rather than normal) also respects literal newlines in headerName
               — used by the "Scheduled debate date" header to force "date" onto its own
               line, so the trailing "?" icon lands next to it instead of wrapping alone. */
            .ag-header-cell-text {
                word-break: normal !important;
                overflow-wrap: normal !important;
                white-space: pre-line !important;
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

            /* Same wrapping treatment for column *group* headers (e.g. "No. of sigs"
               spanning several child columns) — these use a different class than leaf
               headers and aren't covered by wrapHeaderText/autoHeaderHeight. */
            .ag-header-group-text {
                word-break: normal !important;
                overflow-wrap: normal !important;
                white-space: normal !important;
                overflow: visible !important;
                text-overflow: clip !important;
                line-height: 1.3 !important;
            }

            /* Group headers are left-aligned by default; centre this one specifically */
            .centered-group-header .ag-header-group-cell-label {
                justify-content: center;
            }

            /* Merged "select a constituency" placeholder cell shown once, centred
               across the six per-constituency columns, when none is selected. */
            .no-constituency-message,
            .no-constituency-message .agGrid-Markdown {
                color: #C0392B !important;
                text-align: center !important;
                line-height: 1.3 !important;
                width: 100%;
            }

            /* Grey out scheduled-debate dates that have already passed */
            .past-debate-date {
                color: #999 !important;
                background-color: #f2f2f2 !important;
            }

            /* wrapText cells anchor their content to the top of the row by default,
               which leaves "To be considered for debate" sitting at the top of any
               row made taller by a longer petition title in another column — centre
               it vertically instead. */
            .debate-cell {
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }

            /* ag-Grid's default line-height for wrapped cell text is driven by row
               height, which spaces wrapped lines out much more than a normal
               paragraph (same issue as .sig-ratio-cell/.rank-ratio-cell above) —
               tighten it for every wrapText column in this table. */
            #all-petitions-datatable .ag-cell-wrap-text {
                line-height: 1.3 !important;
            }

            /* Non-wrapped cells sit a constant 12px below the cell's top edge
               regardless of row height (a quirk of ag-Grid's default line-height,
               not actual vertical centering). wrapText cells default to flush-top
               instead, so match that same 12px gap here — except .debate-cell,
               which is deliberately centred and would conflict with a fixed offset. */
            #all-petitions-datatable .ag-cell-wrap-text:not(.debate-cell) {
                padding-top: 12px;
            }

            /* Vertical dividers between columns — kept even though resizable is off,
               since the resize handle isn't the only thing that should show a border. */
            #all-petitions-datatable .ag-cell,
            #all-petitions-datatable .ag-header-cell,
            #all-petitions-datatable .ag-header-group-cell {
                border-right: 1px solid #dde2eb !important;
            }

            /* ag-Grid's default cell-focus outline adds a 1px border on all four sides
               of the focused cell. The right edge already has its own divider border
               (set above) with nothing adjacent to double up against, so it stays a
               clean 1px — but the top/left/bottom edges each sit right on top of a
               neighbour's existing border (the row separator, the previous cell's own
               divider), so the focus border stacks with it and reads as thicker. Drop
               the focus border on those three sides entirely so nothing is added on
               top of what's already there; the right edge's permanent divider (still
               !important above) is left alone. */
            #all-petitions-datatable .ag-cell-focus:not(.ag-cell-range-selected):focus-within {
                border-top: none !important;
                border-left: none !important;
                border-bottom: none !important;
                outline: none !important;
            }

            /* No internal grid lines between the six per-constituency columns'
               body cells (they read as one grouped block, not separate data
               columns) while no constituency is selected — but the header row
               keeps its normal dividers between the six column titles. */
            #all-petitions-datatable .span-group-cell {
                border-right: none !important;
            }
            /* Row divider is drawn on .ag-row (spans the full row width), not per
               cell, so it can't be hidden under just these columns by styling the
               cell alone — cover it with a cell that extends 1px past the row's
               own bottom edge, painted in the grid's background colour. */
            #all-petitions-datatable .ag-row {
                overflow: visible;
            }
            #all-petitions-datatable .span-group-cell {
                height: calc(100% + 1px) !important;
                background-color: white;
            }

            /* Show the current-page number in the pagination panel as a text-box, hinting
               it can be clicked and typed into directly (see script below) */
            #all-petitions-datatable .ag-paging-page-summary-panel .ag-paging-number[data-ref="lbCurrent"] {
                display: inline-block;
                box-sizing: border-box;
                width: 38px;
                padding: 1px 0;
                border: 1px solid #adb5bd;
                border-radius: 4px;
                background: #fff;
                cursor: text;
                text-align: center;
                /* ag-Grid's own CSS sets line-height:0 on this element, which collapses
                   its height; override so the box has real height at rest. */
                line-height: 18px;
            }
            #all-petitions-datatable .ag-paging-page-summary-panel .ag-paging-number[data-ref="lbCurrent"] input {
                width: 100%;
                height: 100%;
                box-sizing: border-box;
                text-align: center;
                border: none;
                outline: none;
                background: transparent;
                padding: 0;
                font: inherit;
                color: inherit;
                -moz-appearance: textfield;
            }
            #all-petitions-datatable .ag-paging-page-summary-panel .ag-paging-number[data-ref="lbCurrent"] input::-webkit-outer-spin-button,
            #all-petitions-datatable .ag-paging-page-summary-panel .ag-paging-number[data-ref="lbCurrent"] input::-webkit-inner-spin-button {
                -webkit-appearance: none;
                margin: 0;
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
            // Shared by the page-number-input feature and the pagination-button
            // handler below — jumps the page to the top, instantly (Bootstrap sets
            // html{scroll-behavior:smooth}, which would otherwise turn this into a
            // visible animated scroll).
            function scrollAllPetitionsTableToTop() {
                window.scrollTo({top: 0, left: 0, behavior: 'instant'});
            }

            // Some dropdowns hide their pre-filled value label on reopen (see CSS above);
            // this injects a real placeholder onto the now-empty search input for each one
            // so it reads e.g. "Search for constituency" instead of just being blank.
            (function() {
                var DROPDOWNS = [
                    {id: 'analytics-petition-dropdown', placeholder: 'Search for constituency', width: '210px'},
                    {id: 'debate-date-dropdown', placeholder: 'Select date', width: '150px'},
                    {id: 'upcoming-debate-dropdown', placeholder: 'Select petition', width: '110px'},
                    {id: 'petition-dropdown', placeholder: 'Search for petition', width: '190px'}
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

            // Make the current-page number in the "All Petitions" grid's pagination
            // panel (e.g. the "1" in "Page 1 of 118") clickable and directly typeable,
            // instead of only being able to step through pages with the arrow buttons.
            // ag-Grid redraws this panel on every page change, so a plain click listener
            // on the element itself would get lost — delegate from document instead.
            (function() {
                document.addEventListener('click', function(e) {
                    var target = e.target.closest(
                        '#all-petitions-datatable .ag-paging-page-summary-panel .ag-paging-number[data-ref="lbCurrent"]'
                    );
                    if (!target || target.querySelector('input')) { return; }

                    var api = window.dash_ag_grid.getApi('all-petitions-datatable');
                    if (!api) { return; }

                    var totalPages = api.paginationGetTotalPages();
                    var currentPage = api.paginationGetCurrentPage() + 1;

                    var input = document.createElement('input');
                    input.type = 'number';
                    input.min = 1;
                    input.max = totalPages;
                    input.value = currentPage;

                    target.textContent = '';
                    target.appendChild(input);
                    input.focus();
                    input.select();

                    // If the target page equals the page already showing, ag-Grid treats
                    // paginationGoToPage as a no-op and never redraws this panel, which
                    // would otherwise leave our injected <input> stuck in place with no
                    // page number showing. Restore the plain text ourselves in that case.
                    // `settled` also guards against the blur that fires when this restore
                    // removes the (still-focused) input from the DOM — without it, that
                    // implicit blur would re-run commit() and re-navigate after Escape.
                    var settled = false;
                    function finish(page) {
                        if (settled) { return; }
                        settled = true;
                        // Only a genuine navigation to a *different* page should jump
                        // to the top — re-committing the page already showing (e.g.
                        // clicking in, then clicking away or pressing Escape) must not.
                        if (page !== currentPage) {
                            scrollAllPetitionsTableToTop();
                        }
                        api.paginationGoToPage(page - 1);
                        setTimeout(function() {
                            if (target.contains(input)) {
                                target.textContent = String(api.paginationGetCurrentPage() + 1);
                            }
                        }, 0);
                    }

                    function commit() {
                        var page = parseInt(input.value, 10);
                        finish(isNaN(page) ? currentPage : Math.min(Math.max(page, 1), totalPages));
                    }

                    input.addEventListener('click', function(ev) { ev.stopPropagation(); });
                    input.addEventListener('blur', commit);
                    input.addEventListener('keydown', function(ev) {
                        if (ev.key === 'Enter') {
                            commit();
                        } else if (ev.key === 'Escape') {
                            finish(currentPage);
                        }
                    });
                });
            })();

            // The "All Petitions" grid uses domLayout:'autoHeight', so its own height
            // changes with however many rows land on the new page — that reflow
            // shifts everything below it even though window.scrollY hasn't moved.
            // Rather than try to preserve the user's scroll position across that
            // reflow (which turned out to fight AG Grid's own multi-pass internal
            // row rendering — see git history for that approach and why it was
            // dropped: it either produced a visible jolt, or hiding the grid to mask
            // the jolt made the Next/Prev buttons themselves unclickable, since
            // visibility:hidden removes an element from hit-testing and those buttons
            // are inside the grid), jump to the top of the table on every page change
            // instead, so the new page is always read from its first row.
            //
            // Scoped to clicks on the first/previous/next/last buttons specifically —
            // an earlier version watched the pagination panel's DOM for *any* change
            // (via MutationObserver on the displayed page number's text), which was
            // simple but too broad: clicking the current page number to edit it (see
            // the page-number-input feature above) clears that text as part of
            // swapping in an <input>, and opening the "Page Size" picker mutates the
            // same subtree too — both got misread as "the page changed" and jumped
            // to the top despite the page not actually changing.
            (function() {
                document.addEventListener('click', function(e) {
                    var button = e.target.closest(
                        '#all-petitions-datatable .ag-paging-page-summary-panel [data-ref="btFirst"], ' +
                        '#all-petitions-datatable .ag-paging-page-summary-panel [data-ref="btPrevious"], ' +
                        '#all-petitions-datatable .ag-paging-page-summary-panel [data-ref="btNext"], ' +
                        '#all-petitions-datatable .ag-paging-page-summary-panel [data-ref="btLast"]'
                    );
                    if (!button || button.classList.contains('ag-disabled')) { return; }
                    scrollAllPetitionsTableToTop();
                });
            })();
        </script>
    </body>
</html>
'''


# ── Static components ─────────────────────────────────────

top5_overall_component = render_top5_bars(
    petitions_list[petitions_list['status'] == 'open'],
    'total_signature_count',
    bar_color='#006548',
    border_color='#003f2d'
)

_today = datetime.now().date()
_days_open = (_today - pd.to_datetime(petitions_list['opened_at']).dt.date).apply(lambda d: d.days)

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
    (pd.to_datetime(petitions_list['scheduled_debate_date']) >= pd.Timestamp.now().normalize())
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
        {'label': pd.to_datetime(d).strftime('%d %b %Y'), 'value': d}
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
    dbc.NavLink("Petition Overview", id='tab-2-navlink', active=False),
    dbc.NavLink("All Open Petitions", id='tab-3-navlink', active=False),
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
                                                config={'displayModeBar': False, 'doubleClick': False, 'scrollZoom': False}
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
                    ], className="g-2 align-items-center", style={'marginBottom': '14px'}),

                    dbc.Row([
                        dbc.Col([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6("Total no. of sigs", className="text-muted",
                                                    style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '6px',
                                                           'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                            html.Div(html.H5(id='total-sigs', className="mb-0"),
                                                     style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                        ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '5px 6px'})
                                    ], className="shadow-sm", style={'borderRadius': '10px', 'marginBottom': '8px'}),
                                ], width=12),
                            ], className="g-2"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6("Date opened", className="text-muted",
                                                    style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '6px',
                                                           'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                            html.Div(html.H5(id='date-opened', className="mb-0"),
                                                     style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                        ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '5px 6px'})
                                    ], className="shadow-sm h-100", style={'borderRadius': '10px'}),
                                ], width=6),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6("Scheduled debate date", className="text-muted",
                                                    style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '6px',
                                                           'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                            html.Div(html.H5(id='sch-debate-date', className="mb-0"),
                                                     style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                        ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '5px 6px'})
                                    ], className="shadow-sm h-100", style={'borderRadius': '10px'}),
                                ], width=6),
                            ], className="g-2"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6("No. of sigs in selected constituency", id='petition-constituency-votes-label', className="text-muted",
                                                    style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '6px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                            html.Div(html.H5(id='petition-constituency-votes-box', className="mb-0"),
                                                     style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                        ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '5px 6px'})
                                    ], className="shadow-sm h-100", style={'borderRadius': '10px'}),
                                ], width=6),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                "Ranking based on no. of sigs ",
                                                html.Span("?", id="petition-ranking-info-icon", style={
                                                    'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                                                    'width': '16px', 'height': '16px', 'borderRadius': '50%',
                                                    'border': '1px solid #6c757d', 'color': '#6c757d',
                                                    'fontSize': '11px', 'cursor': 'pointer', 'verticalAlign': 'middle'
                                                }),
                                                dbc.Tooltip(RANK_INFO_TEXT, target="petition-ranking-info-icon", placement="top"),
                                            ], className="text-muted",
                                                    style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '6px', 'textAlign': 'center'}),
                                            html.Div(html.H5(id='petition-ranking-box', className="mb-0"),
                                                     style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                        ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '5px 6px'})
                                    ], className="shadow-sm h-100", style={'borderRadius': '10px'}),
                                ], width=6),
                            ], className="g-2", style={'marginTop': '4px'}),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6("Avg no. of sigs per 1,000 pop", className="text-muted",
                                                    style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '6px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                            html.Div(html.H5(id='petition-sig-per-pop-box', className="mb-0"),
                                                     style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                        ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '5px 6px'})
                                    ], className="shadow-sm h-100", style={'borderRadius': '10px'}),
                                ], width=6),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6([
                                                "Ranking based on avg no. of sigs per 1,000 pop ",
                                                html.Span("?", id="petition-sig-per-pop-rank-info-icon", style={
                                                    'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                                                    'width': '16px', 'height': '16px', 'borderRadius': '50%',
                                                    'border': '1px solid #6c757d', 'color': '#6c757d',
                                                    'fontSize': '11px', 'cursor': 'pointer', 'verticalAlign': 'middle'
                                                }),
                                                dbc.Tooltip(SIG_PER_POP_RANK_INFO_TEXT, target="petition-sig-per-pop-rank-info-icon", placement="top"),
                                            ], className="text-muted",
                                                    style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '6px', 'textAlign': 'center'}),
                                            html.Div(html.H5(id='petition-sig-per-pop-rank-box', className="mb-0"),
                                                     style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                        ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '5px 6px'})
                                    ], className="shadow-sm h-100", style={'borderRadius': '10px'}),
                                ], width=6),
                            ], className="g-2", style={'marginTop': '4px'}),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card(
                                        dbc.CardBody([
                                            html.H6("No. of sigs for all constituencies", className="mb-2 text-center",
                                                    style={'fontSize': '15px', 'fontWeight': 'bold'}),
                                            html.Div(id='petition-top-constituencies-table', style={'flex': '1', 'minHeight': '0', 'overflowY': 'auto'})
                                        ], className="pt-2 pb-2", style={'height': '100%', 'display': 'flex', 'flexDirection': 'column'}),
                                        className="shadow-sm h-100", style={'borderRadius': '10px'}
                                    )
                                ], width=12, style={'height': '100%', 'display': 'flex', 'flexDirection': 'column'}),
                            ], className="g-2", style={'marginTop': '4px', 'flex': '1', 'minHeight': '0'}),
                        ], width=5, style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'}),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    dcc.Graph(
                                        id='petition-histogram',
                                        style={'height': '100%', 'width': '100%'},
                                        config={'displayModeBar': False, 'responsive': True}
                                    )
                                ], className="pt-2 pb-2",
                                    style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'height': '100%', 'minHeight': '0'}),
                                className="shadow-sm h-100", style={'borderRadius': '10px', 'minHeight': '0'}
                            )
                        ], width=7, style={'display': 'flex', 'flexDirection': 'column', 'minHeight': '0'}),
                    ], className="g-2", style={'height': 'calc(100vh - 195px)'})

                ], style={'padding': '20px', 'paddingTop': '14px'})
            ]),

            dcc.Tab(value='tab-3', children=[
                html.Div([

                    dbc.Row([
                        dbc.Col([
                            html.H5("All Open Petitions", className="mb-3 text-center"),
                            html.Div(id='all-petitions-table', children=build_all_petitions_table())
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
    Output('upcoming-debates-histogram', 'config'),
    Input('upcoming-debate-dropdown', 'value'),
    Input('analytics-petition-dropdown', 'value')
)
def update_debate_section(petition_id, PCON24CD):
    no_constituency = html.Span("Select a constituency", style={'color': '#C0392B'})
    no_ranking = html.Span("Rankings only calculated for petitions with at least 10,000 sigs",
                            style={'color': '#C0392B', 'fontSize': '11px'})
    histogram_config = {'displayModeBar': False, 'doubleClick': False, 'scrollZoom': False}
    if petition_id is None:
        blank_fig = go.Figure()
        blank_fig.update_layout(
            dragmode=False,
            xaxis={'visible': False, 'fixedrange': True},
            yaxis={'visible': False, 'fixedrange': True},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin={'r': 0, 't': 0, 'l': 0, 'b': 0}
        )
        # staticPlot removes Plotly's drag layer entirely, so the cursor stays a
        # normal arrow over the blank area instead of showing the drag/pan cursor.
        return "", "No. of sigs in selected constituency", "", "", blank_fig, {**histogram_config, 'staticPlot': True}

    df = petitions_df[petitions_df['petition_id'] == petition_id]

    total_votes = df['total_signature_count'].iloc[0]

    constituency_row = df.loc[df['PCON24CD'] == PCON24CD] if PCON24CD is not None else None

    constituency_votes = constituency_row['signature_count'].iloc[0] if constituency_row is not None else None
    sig_per_pop = constituency_row['sig_per_pop'].iloc[0] if constituency_row is not None else None
    sig_rank = constituency_row['sig_rank_raw'].iloc[0] if constituency_row is not None else None
    sig_percentile = constituency_row['percentile_rank_raw'].iloc[0] if constituency_row is not None else None

    constituency_name = (
        pcon24cds.loc[pcon24cds['PCON24CD'] == PCON24CD, 'constituency_name'].iloc[0]
        if PCON24CD is not None else None
    )
    constituency_label = (
        f"No. of sigs in {constituency_name}"
        if PCON24CD is not None else "No. of sigs in selected constituency"
    )

    fig = render_signature_histogram(df, df['median_signature_count'].iloc[0], constituency_votes, constituency_name)

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
        if pd.notna(sig_rank) else (no_constituency if PCON24CD is None else no_ranking)
    )

    return constituency_votes_str, constituency_label, sig_per_pop_str, ranking_str, fig, histogram_config


# ── All Petitions table ───────────────────────────────────
#
# Patches the already-mounted 'all-petitions-datatable' grid's rowData/columnDefs
# directly (see build_all_petitions_table() above) instead of replacing the whole
# component on every constituency change — see the comment above
# _petitions_display_base for why. prevent_initial_call=True since the layout
# already renders the grid's initial (no constituency selected) state.

@app.callback(
    Output('all-petitions-datatable', 'rowData'),
    Output('all-petitions-datatable', 'columnDefs'),
    Input('analytics-petition-dropdown', 'value'),
    prevent_initial_call=True,
)
def update_all_petitions_table(PCON24CD):
    return _build_all_petitions_rowdata(PCON24CD), _build_all_petitions_columndefs(PCON24CD)


# ── Map tab ───────────────────────────────────────────────

@app.callback(
    Output('petition-histogram', 'figure'),
    Output('total-sigs', 'children'),
    Output('date-opened', 'children'),
    Output('sch-debate-date', 'children'),
    Output('petition-constituency-votes-box', 'children'),
    Output('petition-constituency-votes-label', 'children'),
    Output('petition-sig-per-pop-box', 'children'),
    Output('petition-ranking-box', 'children'),
    Output('petition-sig-per-pop-rank-box', 'children'),
    Output('petition-top-constituencies-table', 'children'),
    Input('petition-dropdown', 'value'),
    Input('analytics-petition-dropdown', 'value')
)
def update_graph(petition_id, PCON24CD):
    callback_start = time.time()

    no_constituency = html.Span("Select a constituency", style={'color': '#C0392B'})
    no_ranking = html.Span("Rankings only calculated for petitions with at least 10,000 sigs",
                            style={'color': '#C0392B', 'fontSize': '11px'})
    paren_style = {'display': 'block', 'marginTop': '6px', 'fontSize': '14px', 'fontWeight': 'normal', 'color': '#888'}

    cached_data = get_petition_data(petition_id)

    df = pd.DataFrame(cached_data, columns=petitions_df.columns)
    print(f"Time to retrieve and process petition data: {time.time() - callback_start:.4f}s")

    total_signatures = df['signature_count'].sum()

    opened_at = df['opened_at'].iloc[0] if 'opened_at' in df.columns else None
    opened_at_str = str(opened_at) if pd.notna(opened_at) else ""

    sch_debate_date = df['scheduled_debate_date'].iloc[0] if 'scheduled_debate_date' in df.columns else None
    debate_date_str = str(sch_debate_date) if pd.notna(sch_debate_date) else "Not scheduled"

    constituency_row = df.loc[df['PCON24CD'] == PCON24CD] if PCON24CD is not None else None
    constituency_votes = constituency_row['signature_count'].iloc[0] if constituency_row is not None else None
    constituency_name = (
        pcon24cds.loc[pcon24cds['PCON24CD'] == PCON24CD, 'constituency_name'].iloc[0]
        if PCON24CD is not None else None
    )
    histogram_fig = render_signature_histogram(df, df['median_signature_count'].iloc[0], constituency_votes, constituency_name)

    sig_per_pop = constituency_row['sig_per_pop'].iloc[0] if constituency_row is not None else None
    sig_rank = constituency_row['sig_rank_raw'].iloc[0] if constituency_row is not None else None
    sig_percentile = constituency_row['percentile_rank_raw'].iloc[0] if constituency_row is not None else None
    sig_per_pop_rank = constituency_row['sig_per_pop_rank'].iloc[0] if constituency_row is not None else None
    sig_per_pop_percentile = constituency_row['percentile_rank_pop'].iloc[0] if constituency_row is not None else None

    constituency_label = (
        f"No. of sigs in {constituency_name}"
        if PCON24CD is not None else "No. of sigs in selected constituency"
    )

    constituency_votes_str = (
        html.Span([
            f"{constituency_votes:,}",
            html.Span(f"({(constituency_votes / total_signatures) * 100:.2f}% of all sigs)", style=paren_style)
        ])
        if constituency_votes is not None else no_constituency
    )
    sig_per_pop_str = f"{sig_per_pop:.2f}" if sig_per_pop is not None else no_constituency
    category = percentile_category(sig_percentile)
    ranking_str = (
        html.Span([f"{int(sig_rank)} of {TOTAL_CONSTITUENCIES}"] + ([html.Span(f"({category})", style=paren_style)] if category else []))
        if pd.notna(sig_rank) else (no_constituency if PCON24CD is None else no_ranking)
    )
    pop_category = percentile_category(sig_per_pop_percentile)
    sig_per_pop_pool = TOTAL_CONSTITUENCIES_SCOTLAND if _is_scotland(PCON24CD) else TOTAL_CONSTITUENCIES_WITH_POP
    sig_per_pop_rank_str = (
        html.Span([f"{int(sig_per_pop_rank)} of {sig_per_pop_pool}"] + ([html.Span(f"({pop_category})", style=paren_style)] if pop_category else []))
        if pd.notna(sig_per_pop_rank) else (no_constituency if PCON24CD is None else no_ranking)
    )

    top_constituencies = df[['constituency_name', 'signature_count']] \
        .sort_values('signature_count', ascending=False) \
        .reset_index(drop=True)

    top_constituencies_table = dag.AgGrid(
        id='petition-top-constituencies-datatable',
        rowData=top_constituencies.to_dict('records'),
        columnDefs=[
            {'field': 'constituency_name', 'headerName': 'Constituency', 'flex': 2, 'minWidth': 180},
            {'field': 'signature_count', 'headerName': 'No. of sigs', 'valueFormatter': _ALL_PETITIONS_NUMBER_FORMAT,
             'flex': 1, 'minWidth': 110, 'cellStyle': {'textAlign': 'center'}, 'headerClass': 'ag-header-center',
             'sort': 'desc'},
        ],
        defaultColDef={'sortable': True, 'resizable': False},
        dashGridOptions={'domLayout': 'autoHeight', 'headerHeight': 28, 'unSortIcon': True},
        className='ag-theme-alpine',
        style={'width': '100%'},
    )

    return (
        histogram_fig,
        f"{total_signatures:,}",
        opened_at_str,
        debate_date_str,
        constituency_votes_str,
        constituency_label,
        sig_per_pop_str,
        ranking_str,
        sig_per_pop_rank_str,
        top_constituencies_table
    )


print(f"[startup] Total time until app is ready to serve: {time.time() - _startup_t0:.2f}s")

if __name__ == '__main__':
    app.run(debug=(ENV == 'local'), port=8051)