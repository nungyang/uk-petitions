####################
#### Setting up ####
####################

# ── Imports ───────────────────────────────────────────────

import time
_startup_t0 = time.time()

import os
import gc
import gzip
import textwrap
from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import boto3
import pandas as pd
import numpy as np
from statsmodels.stats.stattools import medcouple
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

# Created regardless of ENV: local mode falls back to S3 when no local cache
# file matches today/yesterday (see get_petitions_data() etc. below).
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)


# ── S3 loading functions ──────────────────────────────────

def load_csv(filename):
    s3_object = s3_client.get_object(Bucket=bucket, Key=filename)
    body = s3_object['Body'].read()
    if filename.endswith('.gz'):
        body = gzip.decompress(body)
    df = pd.read_csv(BytesIO(body))
    return df


def load_dynamic_csv(base_key):
    """Loads a dynamic-data CSV, preferring the gzip-compressed key the scraper
    now uploads (base_key + '.gz') and falling back to the legacy uncompressed
    key. Only matters for the day of this format switch - it lets the dashboard
    deploy independently of the scraper picking up the new upload format, rather
    than requiring the scraper to run first. Safe to remove once no date within
    the today/yesterday fallback window can still be in the old format."""
    try:
        return load_csv(f'{base_key}.gz')
    except s3_client.exceptions.NoSuchKey:
        return load_csv(base_key)


def save_local_cache(df, filename):
    # Persists an S3 fallback fetch to cached_data/ so the next ENV=local run
    # doesn't have to hit S3 again for the same date.
    cache_path = script_dir / 'cached_data' / filename
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Saved local cache: {cache_path.name}")


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
        print("No local cache found for today or yesterday, falling back to S3...")

    # Same today/yesterday lookup used in production - also the local-mode
    # fallback when no local cache file matches either date. The list and counts
    # files for a given date don't depend on each other, so fetch them
    # concurrently rather than waiting on one full S3 download before starting
    # the next - counts is by far the larger file (megabytes vs kilobytes) and
    # was otherwise blocking on list's download for no reason.
    for delta in [0, 1]:
        date_str = (datetime.now() - timedelta(days=delta)).strftime('%Y%m%d')
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                list_future  = executor.submit(load_dynamic_csv, f'dynamic_data/petitions_list_{date_str}.csv')
                count_future = executor.submit(load_dynamic_csv, f'dynamic_data/petitions_counts_{date_str}.csv')
                petitions_list  = list_future.result()
                petitions_count = count_future.result()
            print(f"Loaded data for {date_str} from S3")
            if ENV == 'local':
                save_local_cache(petitions_list, f'petitions_list_{date_str}.csv')
                save_local_cache(petitions_count, f'petitions_counts_{date_str}.csv')
            return petitions_list, petitions_count
        except s3_client.exceptions.NoSuchKey:
            print(f"No data found for {date_str}, trying previous day...")
    raise FileNotFoundError("No petitions data found in local cache or S3 for today or yesterday")


def get_closed_awaiting_debate_data():
    # Closed petitions currently awaiting a Commons debate (from "daily web scraping
    # closed petitions awaiting debate.py"). Unlike get_petitions_data(), missing data
    # here isn't fatal - the workflow producing it may not have run yet - so this
    # returns (None, None) instead of raising, and callers just skip merging it in.
    if ENV == 'local':
        for delta in [0, 1]:
            date_str = (datetime.now() - timedelta(days=delta)).strftime('%Y%m%d')
            list_path  = script_dir / 'cached_data' / f'closed_awaiting_deb_petitions_list_{date_str}.csv'
            count_path = script_dir / 'cached_data' / f'closed_awaiting_deb_petitions_counts_{date_str}.csv'
            if list_path.exists() and count_path.exists():
                print(f"Loading closed-awaiting-debate petitions data from local cache for {date_str}...")
                return pd.read_csv(list_path), pd.read_csv(count_path)
            print(f"No local cache found for {date_str}, trying previous day...")
        print("No local cache found for today or yesterday, falling back to S3...")

    # Same today/yesterday lookup used in production - also the local-mode
    # fallback when no local cache file matches either date. Fetched concurrently
    # for the same reason as the open-petitions list/counts pair above.
    for delta in [0, 1]:
        date_str = (datetime.now() - timedelta(days=delta)).strftime('%Y%m%d')
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                list_future  = executor.submit(load_dynamic_csv, f'dynamic_data/closed_awaiting_deb_petitions_list_{date_str}.csv')
                count_future = executor.submit(load_dynamic_csv, f'dynamic_data/closed_awaiting_deb_petitions_counts_{date_str}.csv')
                closed_list  = list_future.result()
                closed_count = count_future.result()
            print(f"Loaded closed-awaiting-debate petitions data for {date_str} from S3")
            if ENV == 'local':
                save_local_cache(closed_list, f'closed_awaiting_deb_petitions_list_{date_str}.csv')
                save_local_cache(closed_count, f'closed_awaiting_deb_petitions_counts_{date_str}.csv')
            return closed_list, closed_count
        except s3_client.exceptions.NoSuchKey:
            print(f"No closed-awaiting-debate data found for {date_str}, trying previous day...")
    print("No closed-awaiting-debate petitions data found in local cache or S3 for today or yesterday, skipping.")
    return None, None


def get_electorate_data():
    # Electorate size per constituency, from the House of Commons Library's 2024
    # General Election results — one row per UK constituency (England, Wales,
    # Scotland and Northern Ireland all from the same source), 'ONS ID' matching
    # the PCON24CD codes used everywhere else.
    local_path = script_dir / 'cached_data' / 'HoC_GE2024.csv'
    if ENV == 'local' and local_path.exists():
        print("Loading electorate data from local cache...")
        df = pd.read_csv(local_path)
    else:
        if ENV == 'local':
            print("No local cache found for electorate data, falling back to S3...")
        df = load_csv('static data/HoC_GE2024.csv')
        if ENV == 'local':
            save_local_cache(df, 'HoC_GE2024.csv')
    return df[['ONS ID', 'Electorate']].rename(columns={'ONS ID': 'PCON24CD', 'Electorate': 'electorate'})


@lru_cache(maxsize=128)
def get_petition_data(petition_id):
    start = time.time()
    df = petitions_df[petitions_df['petition_id'] == petition_id]
    print(f"Time to retrieve petition data from cache: {time.time() - start:.4f}s")
    return tuple(df.itertuples(index=False))


# ── Data processing ───────────────────────────────────────

print(f"Environment: {ENV}")

_data_load_t0 = time.time()
print("Loading petitions data, closed-awaiting-debate data and electorate data concurrently...")
with ThreadPoolExecutor(max_workers=3) as executor:
    petitions_future  = executor.submit(get_petitions_data)
    closed_future     = executor.submit(get_closed_awaiting_debate_data)
    electorate_future = executor.submit(get_electorate_data)

    petitions_list, petitions_count = petitions_future.result()
    closed_petitions_list, closed_petitions_count = closed_future.result()
    electorate_df = electorate_future.result()

if closed_petitions_list is not None:
    # Merged in at the source so the cross-join/ranking pipeline below treats them
    # like any other petition_id. Everywhere that should stay open-petitions-only
    # (top 5s, All Open Petitions, ...) filters status == 'open' explicitly.
    #
    # A petition can appear in both pulls (e.g. it closed today but the open pull
    # fell back to yesterday's file) - keep the closed-awaiting-debate copy, since
    # it's the more current one, and drop the stale 'open' duplicate.
    petitions_list = pd.concat([petitions_list, closed_petitions_list], ignore_index=True) \
        .drop_duplicates(subset='petition_id', keep='last')
    petitions_count = pd.concat([petitions_count, closed_petitions_count], ignore_index=True) \
        .drop_duplicates(subset=['petition_id', 'PCON24CD'], keep='last')

print("Done loading data.")
print(f"[startup] S3/local data download: {time.time() - _data_load_t0:.2f}s")

_merge_t0 = time.time()
# Data pull has missing rows if value should actually be 0 so making sure they get filled with 0
petition_ids = petitions_list[['petition_id']].drop_duplicates()
pcon24cds = petitions_count[['PCON24CD', 'constituency_name']].drop_duplicates()
TOTAL_CONSTITUENCIES = len(pcon24cds)

skeleton_df = petition_ids.merge(pcon24cds, how='cross')

petitions_count = skeleton_df.merge(petitions_count.drop(columns=['constituency_name']), on = ['petition_id', 'PCON24CD'], how = 'left')
petitions_count['signature_count'] = petitions_count['signature_count'].fillna(0).astype(int)

# output_path = script_dir / "cached_data" / "file_to_check.csv"
# petitions_count.to_csv(output_path, index=False)

# Adding data on electorate size
petitions_count = petitions_count.merge(electorate_df[['PCON24CD', 'electorate']], on= ['PCON24CD'], how='left')

# Adding column on no. of sigs as a proportion of electorate
petitions_count['sig_prop_electorate'] = (petitions_count['signature_count'] / petitions_count['electorate']) * 100

# Adding rank
petitions_count['sig_prop_electorate_rank'] = petitions_count.groupby('petition_id')['sig_prop_electorate'].rank(ascending=False, method='min')
petitions_count['percentile_rank_electorate'] = (100 - (petitions_count.groupby('petition_id')['sig_prop_electorate'].rank(pct=True) * 100)).round(1)

# Working out median count for each petition
median_counts = petitions_count.groupby('petition_id')['signature_count'].median().reset_index()
median_counts.columns = ['petition_id', 'median_signature_count']

# Adding median counts to petitions list
petitions_list = petitions_list.merge(median_counts, on='petition_id', how='left')

# Merging petitions count to petitions list
petitions_df = petitions_list.merge(petitions_count, on='petition_id', how='left')

# petitions_count/skeleton_df/electorate_df/petition_ids/median_counts were only ever
# stepping stones toward petitions_df — everything they held is now merged in,
# and nothing downstream reads them again, so free the memory rather than
# leaving them resident as unused module-level globals for the app's lifetime.
del petitions_count, skeleton_df, electorate_df, petition_ids, median_counts
gc.collect()

# If total count is less than 10,000, then remove ranking
petitions_df.loc[petitions_df['total_signature_count'] <= 10000, 'percentile_rank_electorate'] = np.nan
petitions_df.loc[petitions_df['total_signature_count'] <= 10000, 'sig_prop_electorate_rank'] = np.nan

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

# Displays a rank (e.g. sig_prop_electorate_rank) as "N of {TOTAL_CONSTITUENCIES}"
# while the underlying cell value stays numeric, so ascending/descending sort works.
RANK_DISPLAY_FORMAT = {'function': (
    f"params.value == null || params.value === 0 ? '' : params.value + ' of {TOTAL_CONSTITUENCIES} constituencies'"
)}

# Explains via a "?" tooltip why a petition's signature-rate ranking can be blank —
# it's suppressed for petitions with under 10,000 signatures (see the
# total_signature_count <= 10000 filtering above).
RANK_INFO_TEXT = "Ranking only shows for petitions with 10,000 or more signatures"

VIEW_PETITION_BTN_STYLE = {
    'position': 'absolute', 'top': '14px', 'right': '18px',
    'fontSize': '12px', 'fontWeight': 'bold', 'color': 'white',
    'border': '1px solid #373151', 'borderRadius': '6px',
    'padding': '4px 10px', 'textDecoration': 'none',
    'backgroundColor': '#373151'
}


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


# Explains how "signature rate" is calculated, wherever it's shown as a stat-box value
SIGNATURE_RATE_LABEL = "Signature rate (% of electorate)"
SIGNATURE_RATE_INFO_TEXT = "Signature rate is calculated as (number of signatures)/(size of electorate)* 100"

# Same pattern for the "All data" table's "Signature rate" header.
SIGNATURE_RATE_HEADER_INFO_ICON_ID = 'signature-rate-header-info-icon'
SIGNATURE_RATE_HEADER_TEMPLATE = make_header_info_icon_template(SIGNATURE_RATE_HEADER_INFO_ICON_ID)

TOP5_PERCENT_TITLE_INFO_ICON_ID = 'top5-percent-title-info-icon'


def top5_percent_title_text(text):
    return html.Span([
        text + " ",
        html.Span("?", id=TOP5_PERCENT_TITLE_INFO_ICON_ID, style={
            'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
            'width': '16px', 'height': '16px', 'flexShrink': '0', 'borderRadius': '50%',
            'border': '1px solid #6c757d', 'color': '#6c757d',
            'fontSize': '11px', 'cursor': 'pointer', 'verticalAlign': 'middle'
        }),
        dbc.Tooltip(SIGNATURE_RATE_INFO_TEXT, target=TOP5_PERCENT_TITLE_INFO_ICON_ID, placement='top'),
    ])

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


def calculate_upperfence(data):
    """Skew-adjusted boxplot upper fence (medcouple method) used to flag outlier
    constituencies on the Petition Overview table."""
    data = np.asarray(sorted(data))
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    mc = medcouple(data)

    if mc >= 0:
        upper_fence = q3 + 1.5 * np.exp(3 * mc) * iqr
    else:
        upper_fence = q3 + 1.5 * np.exp(4 * mc) * iqr

    return upper_fence


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
    df = petitions_list[petitions_list['status'] == 'open'].copy()
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
        'petition_id', 'signature_count', 'percentile_rank_electorate',
        'sig_prop_electorate', 'sig_prop_electorate_rank'
    ]].drop_duplicates(subset='petition_id').copy()
    constituency_sigs['petition_id'] = constituency_sigs['petition_id'].astype(str)

    df = _petitions_display_base(today).merge(constituency_sigs, on='petition_id', how='left')

    if PCON24CD is not None:
        df['signature_count'] = df['signature_count'].astype(int)
        # sig_prop_electorate_rank is NaN for petitions with <= 10,000 total signatures
        # (rank suppressed above). astype(int) can't hold NaN, so cast element-wise and
        # leave those as None instead of crashing.
        df['sig_prop_electorate_rank'] = df['sig_prop_electorate_rank'].apply(lambda x: int(x) if pd.notna(x) else None)
    df['percentile_rank_electorate'] = df['percentile_rank_electorate'].round(1)

    table_df = df[[
        'petition_title_link',
        'opened_at',
        'months_open_rank',
        'total_signature_count',
        'signature_count',
        'sig_prop_electorate',
        'sig_prop_electorate_rank',
        'percentile_rank_electorate',
        'debate_display',
        'debate_sort_key',
        'is_past_debate'
    ]].sort_values('total_signature_count', ascending=False)

    return table_df.to_dict('records')


_ALL_PETITIONS_PAGE_SIZE = 20

# When no constituency is selected, a row near the top of each page carries the
# merged placeholder message (colSpan across all 4 per-constituency columns);
# every other row's cell in that column renders blank. dash_ag_grid compiles
# colSpan/valueGetter as an expression-bodied arrow function (params) => (CODE) —
# no statements/var/return allowed — so this has to be one composed expression.
_ALL_PETITIONS_NEAR_TOP_OFFSET = 1
_all_petitions_page_start = f"(Math.floor(params.node.rowIndex / {_ALL_PETITIONS_PAGE_SIZE}) * {_ALL_PETITIONS_PAGE_SIZE})"
_all_petitions_target_row = f"({_all_petitions_page_start} + {_ALL_PETITIONS_NEAR_TOP_OFFSET})"
_ALL_PETITIONS_IS_TARGET_ROW = f"(params.node.rowIndex === {_all_petitions_target_row})"

_ALL_PETITIONS_NUMBER_FORMAT = {'function': "d3.format(',')(params.value)"}


def _build_all_petitions_columndefs(PCON24CD):
    # When no constituency is selected, the four per-constituency columns lose their
    # internal grid lines (both the vertical divider between them and the row line
    # under them) so they read as one blank panel instead of four empty columns;
    # once a constituency is picked they get their normal gridlines back.
    SPAN_GROUP_CELL_CLASS = 'span-group-cell' if PCON24CD is None else ''
    SPAN_GROUP_HEADER_CLASS = 'span-group-header' if PCON24CD is None else ''

    signature_count_coldef = (
        {'field': 'signature_count', 'headerName': 'No. of sigs in\nconstituency',
         'cellRenderer': 'markdown', 'cellClass': f'no-constituency-message {SPAN_GROUP_CELL_CLASS}',
         'headerClass': SPAN_GROUP_HEADER_CLASS, 'sortable': True,
         'valueGetter': {'function': f"{_ALL_PETITIONS_IS_TARGET_ROW} ? 'Select a constituency  \\n(see top right)' : ''"},
         'colSpan': {'function': f"{_ALL_PETITIONS_IS_TARGET_ROW} ? 4 : 1"},
         'flex': 0.9, 'minWidth': 155}
        if PCON24CD is None else
        {'field': 'signature_count', 'headerName': 'No. of sigs in\nconstituency',
         'valueFormatter': _ALL_PETITIONS_NUMBER_FORMAT, 'flex': 0.9, 'minWidth': 155}
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
        {'field': 'debate_sort_key', 'headerName': 'Scheduled debate\ndate', 'flex': 0.65, 'minWidth': 140,
         'valueFormatter': {'function': "params.data.debate_display || ''"},
         'cellClass': {'function': "'debate-cell' + (params.data.is_past_debate ? ' past-debate-date' : '')"},
         'wrapText': True, 'autoHeight': True,
         'headerComponentParams': {'template': SCHEDULED_DEBATE_HEADER_TEMPLATE}},
        {'field': 'total_signature_count', 'headerName': 'Total no.\nof sigs',
            'valueFormatter': _ALL_PETITIONS_NUMBER_FORMAT, 'flex': 0.7, 'minWidth': 125},
        signature_count_coldef,
        {'field': 'sig_prop_electorate', 'headerName': 'Signature rate', 'flex': 0.75, 'minWidth': 125,
         'valueFormatter': {'function': "params.value == null ? '' : params.value.toFixed(2) + '%'"},
         'cellClass': SPAN_GROUP_CELL_CLASS, 'headerClass': f'{SPAN_GROUP_HEADER_CLASS} ag-header-center'.strip(),
         'headerComponentParams': {'template': SIGNATURE_RATE_HEADER_TEMPLATE}},
        {'field': 'sig_prop_electorate_rank', 'headerName': 'Ranking based on sig rate',
         'valueFormatter': RANK_DISPLAY_FORMAT, 'cellClass': SPAN_GROUP_CELL_CLASS,
         'headerClass': SPAN_GROUP_HEADER_CLASS, 'flex': 1, 'minWidth': 170,
         'wrapText': True, 'autoHeight': True},
        {'field': 'percentile_rank_electorate', 'headerName': 'Ranking as percentile', 'flex': 0.6, 'minWidth': 130,
         'cellClass': SPAN_GROUP_CELL_CLASS, 'headerClass': SPAN_GROUP_HEADER_CLASS,
         'valueFormatter': PERCENTILE_CATEGORY_FORMAT,
         'wrapText': True, 'autoHeight': True},
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
                         'unSortIcon': True, 'groupHeaderHeight': 56,
                         # autoHeaderHeight (below) measures the wrapped header text's real
                         # height via a ResizeObserver that only fires *after* the header
                         # first paints at ag-Grid's small built-in default — so on every
                         # mount the header visibly grows a beat after the rest of the grid
                         # already looks settled. Seeding the leaf header row's height with
                         # the value that measurement always converges to (given this
                         # column set's fixed minWidths, which keep wrapping at 2 lines
                         # regardless of viewport width) makes the first paint already
                         # correct, so there's nothing left to visibly snap into place.
                         # autoHeaderHeight stays on as a safety net if that ever changes.
                         'headerHeight': 62,
                         'enableCellSpan': True,
                         # ag-Grid defaults animateRows to True, which slides each row into
                         # its position via a CSS transform transition whenever the grid
                         # (re)renders - including the very first render. Combined with
                         # domLayout='autoHeight' that reads as the whole table panning
                         # open from top to bottom on mount. Not wanted here.
                         'animateRows': False},
        dangerously_allow_code=True,
        className='ag-theme-alpine',
        style={'width': '100%'},
    )

    return html.Div([
        table,
        dbc.Tooltip(SCHEDULED_DEBATE_INFO_TEXT, target=SCHEDULED_DEBATE_INFO_ICON_ID, placement='top'),
        dbc.Tooltip(MONTHS_OPEN_INFO_TEXT, target=MONTHS_OPEN_INFO_ICON_ID, placement='top'),
        dbc.Tooltip(SIGNATURE_RATE_INFO_TEXT, target=SIGNATURE_RATE_HEADER_INFO_ICON_ID, placement='top'),
    ], style={'width': '100%'})


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


def render_signature_histogram(df, median_value, highlight_value=None, constituency_name=None,
                                value_col='signature_count', x_axis_title='Number of signatures',
                                bin_unit_label='no of sig', value_fmt=None, discrete=True,
                                tick_format=',d', tick_suffix='', hide_zero_tick=False):
    """Histogram of value_col across constituencies for one petition, with a dotted
    vertical line at median_value and, if highlight_value is given, the bin
    containing it picked out in a lighter shade of green.

    discrete=True (the default, for raw signature counts) keeps bin edges on whole
    numbers. Continuous metrics (e.g. signatures as a proportion of electorate) pass
    discrete=False to bin the observed range directly instead.
    """
    values = df[value_col]
    value_fmt = value_fmt or (lambda v: f"{int(round(v)):,}")
    max_val = values.max()
    if discrete:
        max_val = int(max_val)
        bin_width = max(1, -(-max_val // 30))  # ceil(max_val / 30), so bin edges land on whole numbers
        num_bins = max(1, -(-max_val // bin_width))  # ceil(max_val / bin_width)
        bin_edges = np.arange(num_bins + 1) * bin_width
    else:
        num_bins = 30
        bin_edges = np.linspace(0, max_val if max_val > 0 else 1, num_bins + 1)
    counts, bin_edges = np.histogram(values, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    bin_labels = [
        f"{value_fmt(bin_edges[i])} < {bin_unit_label} ≤ {value_fmt(bin_edges[i + 1])}"
        for i in range(len(bin_edges) - 1)
    ]

    bar_colors = ['#006548'] * len(counts)
    if highlight_value is not None:
        highlight_bin = np.searchsorted(bin_edges, highlight_value, side='right') - 1
        highlight_bin = min(max(highlight_bin, 0), len(counts) - 1)
        bar_colors[highlight_bin] = '#40a583'

    count_labels = [
        f"{c} constituency" if c == 1 else f"{c} constituencies"
        for c in counts
    ]

    # Bin index of each row, computed the same way as highlight_bin above, so it's
    # guaranteed consistent with how np.histogram assigned rows to `counts`.
    bin_indices = np.searchsorted(bin_edges, values.values, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, len(counts) - 1)
    names_by_bin = (
        pd.Series(df['constituency_name'].values, index=bin_indices)
        .groupby(level=0).apply(sorted)
    )

    constituency_lists = []
    for i, c in enumerate(counts):
        if 0 < c <= 5:
            names = names_by_bin.get(i, [])
            header = "The constituency in this interval is:" if c == 1 else "Constituencies in this interval are:"
            bullets = "<br>&nbsp; &nbsp;".join(f"• {n} &nbsp; &nbsp;" for n in names)
            constituency_lists.append(f"<br>&nbsp;<br>&nbsp; &nbsp;{header} &nbsp; &nbsp;<br>&nbsp; &nbsp;{bullets}")
        else:
            constituency_lists.append("")

    fig = go.Figure(go.Bar(
        x=bin_centers, y=counts, width=bin_widths, marker_color=bar_colors,
        text=bin_labels, textposition='none',
        customdata=list(zip(count_labels, constituency_lists)),
        hovertemplate=(
            '&nbsp;<br>&nbsp; &nbsp;%{customdata[0]} &nbsp; &nbsp;<br>&nbsp; &nbsp;%{text} &nbsp; &nbsp;'
            '%{customdata[1]}<br>&nbsp;<extra></extra>'
        ),
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

    dtick = max(1, round(max_val / 6)) if discrete else (max_val / 6 if max_val > 0 else 1)
    if hide_zero_tick:
        # Explicit tick positions/labels (rather than tick0/dtick) so the "0" label
        # specifically can be blanked out while every other tick stays labelled.
        tick_positions = np.arange(0, bin_edges[-1] + dtick / 2, dtick)
        tick_kwargs = dict(
            tickmode='array', tickvals=tick_positions,
            ticktext=['' if abs(t) < 1e-9 else value_fmt(t) for t in tick_positions]
        )
    else:
        tick_kwargs = dict(tick0=0, dtick=dtick, tickformat=tick_format, ticksuffix=tick_suffix)
    fig.update_layout(
        hovermode='x',
        dragmode=False,
        xaxis_title=x_axis_title,
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
                   **tick_kwargs),
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
            /* The "(N of M)" line is a separate markdown paragraph (blank line in the
               source) - react-markdown renders paragraphs as <div> here (see the
               'p: "div"' component override), so it's a sibling div, not a <p> -
               so it can be styled independently, greyed out and smaller like the
               "(Top N%)" category line elsewhere (see paren_style). */
            .sig-ratio-cell .agGrid-Markdown div:last-child {
                color: #888;
                font-size: 11px;
                margin-top: 4px;
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
                line-height: 1.3;
            }

            /* AG-Grid's own CSS gives .ag-center-cols-viewport a min-height: 100%, which
               (in domLayout=autoHeight mode, where the viewport's own height is derived
               from its content) creates a circular/self-inflating height once wrapText
               row heights come in shorter than the grid's initial estimate — leaving a
               large empty gap below the last row. Break that feedback loop here. */
            #top5-percent-datatable .ag-center-cols-viewport,
            #top5-percent-datatable .ag-center-cols-container,
            #all-petitions-datatable .ag-center-cols-viewport,
            #all-petitions-datatable .ag-center-cols-container,
            #up-and-coming-datatable .ag-center-cols-viewport,
            #up-and-coming-datatable .ag-center-cols-container {
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
            #tab-1-navlink, #tab-2-navlink, #tab-3-navlink, #tab-4-navlink {
                cursor: pointer;
            }

            /* Force the longer nav labels onto two lines so every top-nav link
               reads at a consistent width instead of stretching the banner. */
            .page-navlink-wrap {
                white-space: normal !important;
                width: 130px;
                text-align: center;
                line-height: 1.2;
            }

            /* Native dcc.Tabs header is replaced by the nav in the top banner; hide it
               but keep dcc.Tabs itself so its tab-switching/content mount-unmount logic
               (and the Map tab's plotly graph sizing) still works. */
            .tab-container {
                display: none !important;
            }

            /* dcc.Tabs' own wrapper sets overflow: hidden (for the hidden tab header's
               horizontal scrolling), which silently breaks position: sticky for anything
               inside it - e.g. the About page's nav pane. Safe to relax since the header
               it was protecting is hidden above anyway. */
            .tab-parent {
                overflow: visible !important;
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

            /* Horizontally centre the debate-date cell's (possibly wrapped) text,
               same as every other column, while still anchoring it to the top of
               the row like the rest of the table. */
            .debate-cell {
                display: flex !important;
                align-items: flex-start !important;
                justify-content: center !important;
                padding-top: 12px !important;
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
               instead, so match that same 12px gap here (.debate-cell gets its own
               copy of this above, since it needs !important to override its flex
               centring). */
            #all-petitions-datatable .ag-cell-wrap-text {
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
            /* Body text on the About page shouldn't show the text-selection (I-beam)
               cursor; hyperlinks should still show the pointer cursor. */
            #about-page-content, #about-page-content * {
                cursor: default;
            }
            #about-page-content a {
                cursor: pointer;
                color: #1155CC;
            }
            #about-page-content a:visited {
                color: #6B3FA0;
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
        dashGridOptions={'domLayout': 'autoHeight', 'animateRows': False},
        dangerously_allow_code=True,
        className='ag-theme-alpine',
        style={'width': '100%'},
    )

# ── Dropdowns ─────────────────────────────────────────────

petition_options = petitions_list[['petition_id', 'petition_title', 'total_signature_count']] \
    .sort_values('total_signature_count', ascending=False).copy()
petition_options['petition_label'] = petition_options.apply(
    lambda r: f"{r['petition_title']} ({r['total_signature_count']:,} signatures)",
    axis=1
)

petition_dropdown = dcc.Dropdown(
    id='petition-dropdown',
    options=[
        {'label': row['petition_label'], 'value': row['petition_id']}
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
        for _, row in pcon24cds.sort_values('constituency_name').iterrows()
    ],
    placeholder='Select a constituency',
    clearable=False,
    style={'width': '320px'},
    persistence=True,
    persistence_type='local',
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
    dbc.NavLink("About", id='tab-4-navlink', active=False),
    dbc.NavLink("Constituency Overview", id='tab-1-navlink', active=True, className="page-navlink-wrap"),
    dbc.NavLink("Petition Overview", id='tab-2-navlink', active=False, className="page-navlink-wrap"),
    dbc.NavLink("All Open Petitions", id='tab-3-navlink', active=False, className="page-navlink-wrap"),
], pills=True, className="gap-5 align-items-center")

banner = dbc.Navbar(
    dbc.Container([
        html.Img(src=app.get_asset_url('Logo.png'), style={'height': '68px'}),
        page_nav,
        dbc.Row([
            dbc.Col(html.Label("Constituency:", className="text-white mb-0 me-2"), width="auto"),
            dbc.Col(constituency_dropdown, width="auto"),
        ], align="center", className="g-2 flex-nowrap"),
    ], fluid=True, style={'paddingLeft': '34px', 'paddingRight': '32px'}),
    color="#373151",
    dark=True,
)


# ── App layout ────────────────────────────────────────────

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
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
                                ], className="pt-3 pb-2"),
                                className="shadow-sm h-100", style={'borderRadius': '14px'}
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
                                ], className="pt-3 pb-2"),
                                className="shadow-sm h-100", style={'borderRadius': '14px'}
                            )
                        ], style={'flex': '0 0 31%', 'maxWidth': '31%'}),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5(
                                        id='top5-percent-title', className="mb-1 text-center",
                                        style={'minHeight': '40px', 'display': 'flex', 'alignItems': 'flex-start', 'justifyContent': 'center', 'lineHeight': '1.5'}
                                    ),
                                    html.Div(id='top5-percent-table', style={'paddingTop': '2px'})
                                ], className="pt-3 pb-2"),
                                className="shadow-sm h-100", style={'borderRadius': '14px'}
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
                                ], className="pt-3 pb-2", style={'paddingLeft': '8px', 'paddingRight': '8px'}),
                                className="shadow-sm h-100", style={'borderRadius': '14px'}
                            )
                        ], style={'flex': '0 0 39%', 'maxWidth': '39%'}),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.A(
                                        "View petition ↗",
                                        id='view-petition-link-btn',
                                        href='#',
                                        target='_blank',
                                        style={**VIEW_PETITION_BTN_STYLE, 'display': 'none'}
                                    ),
                                    html.Div([
                                        html.H5("Upcoming debate(s) on", className="mb-0 me-2"),
                                        debate_date_dropdown
                                    ], className="mb-2 d-flex align-items-center justify-content-center"),
                                    html.Div(upcoming_debate_dropdown, className="mb-4"),
                                    html.Div(
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Card([
                                                    dbc.CardBody([
                                                        html.H6("Total no. of sigs", className="text-muted",
                                                                style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '12px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                                        html.Div(html.H5(id='debate-constituency-votes-box', className="mb-0"),
                                                                 style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                                    ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '8px 8px'})
                                                ], className="shadow-sm mb-2", style={'borderRadius': '14px'}),
                                                dbc.Card([
                                                    dbc.CardBody([
                                                        html.H6(
                                                            html.Span([
                                                                SIGNATURE_RATE_LABEL + " ",
                                                                html.Span("?", id="debate-signature-rate-info-icon", style={
                                                                    'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                                                                    'width': '16px', 'height': '16px', 'flexShrink': '0', 'borderRadius': '50%',
                                                                    'border': '1px solid #6c757d', 'color': '#6c757d',
                                                                    'fontSize': '11px', 'cursor': 'pointer', 'verticalAlign': 'middle'
                                                                }),
                                                                dbc.Tooltip(SIGNATURE_RATE_INFO_TEXT, target="debate-signature-rate-info-icon", placement="top"),
                                                            ]),
                                                            className="text-muted",
                                                            style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '12px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                                        html.Div(html.H5(id='debate-sig-prop-electorate-box', className="mb-0"),
                                                                 style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                                    ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '8px 8px'})
                                                ], className="shadow-sm mb-2", style={'borderRadius': '14px'}),
                                                dbc.Card([
                                                    dbc.CardBody([
                                                        html.H6("Ranking based on signature rate", className="text-muted",
                                                                style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '12px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                                        html.Div(html.H5(id='debate-ranking-box', className="mb-0"),
                                                                 style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                                    ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '8px 8px'})
                                                ], className="shadow-sm", style={'borderRadius': '14px'}),
                                            ], style={'flex': '0 0 22%', 'maxWidth': '22%'}),
                                            dbc.Col([
                                                dcc.Graph(
                                                    id='upcoming-debates-histogram',
                                                    style={'height': '350px'},
                                                    config={'displayModeBar': False, 'doubleClick': False, 'scrollZoom': False}
                                                )
                                            ], style={'flex': '0 0 78%', 'maxWidth': '78%'}),
                                        ], className="g-2 mb-2", style={'width': '100%'}),
                                        style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'width': '100%'}
                                    ),
                                ], className="pt-3 pb-2", style={'position': 'relative', 'display': 'flex', 'flexDirection': 'column', 'height': '100%'}),
                                className="shadow-sm h-100", style={'borderRadius': '14px'}
                            )
                        ], style={'flex': '0 0 61%', 'maxWidth': '61%'})
                    ], className="g-2 mt-2"),

                ], style={'padding': '20px'})
            ]),

            dcc.Tab(value='tab-2', children=[
                html.Div([

                    html.A(
                        "View petition ↗",
                        id='view-petition-link-btn-2',
                        href='#',
                        target='_blank',
                        style={**VIEW_PETITION_BTN_STYLE, 'display': 'none'}
                    ),

                    dbc.Row([
                        dbc.Col(
                            html.Label("Select a Petition:", className="fw-bold mb-0"),
                            width="auto", className="d-flex align-items-center"
                        ),
                        dbc.Col(petition_dropdown, width="auto", style={'width': '780px', 'maxWidth': '780px'}),
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
                                            html.H6(
                                                html.Span([
                                                    "Median signature rate", html.Br(),
                                                    "(% of electorate) ",
                                                    html.Span("?", id="petition-median-sig-rate-info-icon", style={
                                                        'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                                                        'width': '16px', 'height': '16px', 'flexShrink': '0', 'borderRadius': '50%',
                                                        'border': '1px solid #6c757d', 'color': '#6c757d',
                                                        'fontSize': '11px', 'cursor': 'pointer', 'verticalAlign': 'middle'
                                                    }),
                                                    dbc.Tooltip(SIGNATURE_RATE_INFO_TEXT, target="petition-median-sig-rate-info-icon", placement="top"),
                                                ]),
                                                className="text-muted",
                                                style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '6px', 'height': '34px',
                                                       'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'textAlign': 'center'}),
                                            html.Div(html.H5(id='petition-median-sig-rate-box', className="mb-0"),
                                                     style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                        ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '8px 6px'})
                                    ], className="shadow-sm h-100", style={'borderRadius': '10px'}),
                                ], width=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6(
                                                html.Span([
                                                    SIGNATURE_RATE_LABEL + " ",
                                                    html.Span("?", id="petition-signature-rate-info-icon", style={
                                                        'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                                                        'width': '16px', 'height': '16px', 'flexShrink': '0', 'borderRadius': '50%',
                                                        'border': '1px solid #6c757d', 'color': '#6c757d',
                                                        'fontSize': '11px', 'cursor': 'pointer', 'verticalAlign': 'middle'
                                                    }),
                                                    dbc.Tooltip(SIGNATURE_RATE_INFO_TEXT, target="petition-signature-rate-info-icon", placement="top"),
                                                ]),
                                                className="text-muted",
                                                style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '6px', 'height': '34px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                                            html.Div(html.H5(id='petition-sig-prop-electorate-box', className="mb-0"),
                                                     style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                        ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '8px 6px'})
                                    ], className="shadow-sm h-100", style={'borderRadius': '10px'}),
                                ], width=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6(
                                                html.Span([
                                                    "Ranking based on", html.Br(), "signature rate ",
                                                    html.Span("?", id="petition-sig-prop-electorate-rank-info-icon", style={
                                                        'display': 'inline-flex', 'alignItems': 'center', 'justifyContent': 'center',
                                                        'width': '16px', 'height': '16px', 'flexShrink': '0', 'borderRadius': '50%',
                                                        'border': '1px solid #6c757d', 'color': '#6c757d',
                                                        'fontSize': '11px', 'cursor': 'pointer', 'verticalAlign': 'middle'
                                                    }),
                                                    dbc.Tooltip(RANK_INFO_TEXT, target="petition-sig-prop-electorate-rank-info-icon", placement="top"),
                                                ]),
                                                className="text-muted",
                                                style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '6px', 'height': '34px',
                                                       'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'textAlign': 'center'}),
                                            html.Div(html.H5(id='petition-sig-prop-electorate-rank-box', className="mb-0"),
                                                     style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
                                        ], className="text-center", style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'padding': '8px 6px'})
                                    ], className="shadow-sm h-100", style={'borderRadius': '10px'}),
                                ], width=4),
                            ], className="g-2", style={'marginTop': '4px'}),
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id='petition-top-constituencies-table', style={'flex': '1', 'minHeight': '0'})
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

                ], style={'padding': '20px', 'paddingTop': '14px', 'position': 'relative'})
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
            ]),

            dcc.Tab(value='tab-4', children=[
                html.Div([

                    dbc.Row([
                        dbc.Col(
                            html.Div([
                                html.A(
                                    "Overview", href="#overview",
                                    className="d-block mb-2"
                                ),
                                html.A(
                                    "Background", href="#background",
                                    className="d-block mb-2"
                                ),
                                html.A(
                                    "Technical notes", href="#technical-notes",
                                    className="d-block mb-2"
                                ),
                                html.A(
                                    "Shortcomings to petition data",
                                    href="#shortcomings",
                                    className="d-block mb-2 ms-3",
                                    style={'fontSize': '13px'}
                                ),
                                html.A(
                                    "Figures may differ to official site",
                                    href="#figures-differ",
                                    className="d-block mb-2 ms-3",
                                    style={'fontSize': '13px'}
                                ),
                                html.A(
                                    "Reasons for using electorate",
                                    href="#electorate",
                                    className="d-block mb-2 ms-3",
                                    style={'fontSize': '13px'}
                                ),
                                html.A(
                                    "Contact", href="#contact",
                                    className="d-block mb-2"
                                ),
                                html.A(
                                    "Licence", href="#licence",
                                    className="d-block mb-2"
                                ),
                                html.A(
                                    "↑ Back to top",
                                    href="#about-page-content",
                                    className="d-block mt-4"
                                ),
                            ], className="sticky-top", style={
                                'top': '20px', 'marginTop': '16px', 'padding': '20px',
                                'border': '1px solid rgba(0, 0, 0, 0.176)', 'borderRadius': '14px',
                                'backgroundColor': '#fff'
                            }),
                            id='about-nav-pane', width=3
                        ),
                        dbc.Col([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5("Overview", id="overview", className="mb-3", style={'textDecoration': 'underline', 'fontSize': '26px'}),
                                    html.P(
                                        "The UK Petitions Dashboard has been designed as a tool to help MPs "
                                        "and their staff make better use of publicly available data on UK "
                                        "e-petitions, by presenting key statistics in a more accessible way. "
                                        "It is run independently of the UK Government and Parliament."
                                    ),
                                ]),
                                className="mb-4", style={'borderRadius': '14px', 'border': 'none'}
                            ),
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5("Background", id="background", className="mb-3", style={'textDecoration': 'underline', 'fontSize': '26px'}),
                                    html.P([
                                        "The ",
                                        html.A(
                                            "UK Government and Parliament Petitions",
                                            href="https://petition.parliament.uk/",
                                            target="_blank",
                                        ),
                                        " website hosts numerous petitions, each calling for a change in "
                                        "Government policy or legislation. Unlike other petition websites, "
                                        "such as Change.org, any petition that has over 10,000 signatures "
                                        "receives a written response from the Government, while those that "
                                        "have over 100,000 signatures are considered for debate in Parliament."
                                    ]),
                                    html.P(
                                        "The UK Petitions website hosts data on the number of signatures for "
                                        "each petition, including a breakdown by constituency. That data can "
                                        "be accessed by navigating to each individual petition page, but there "
                                        "is currently no easy way of gaining an overview of petition data, "
                                        "especially at the constituency level."
                                    ),
                                    html.P(
                                        "The aim of UK Petition Analytics is to make that data more accessible "
                                        "for MPs and their staff. Petition data serves as a useful temperature "
                                        "check to gauge how pertinent various issues are in any given "
                                        "constituency and how that compares to another. As noted above, those "
                                        "with more than 100,000 signatures are considered for debate, which "
                                        "means an oversight of petitions can help MPs better decide how to "
                                        "organise their time. Given that petition data is so readily "
                                        "available, it would be a missed opportunity not to make better use "
                                        "of it."
                                    ),
                                ]),
                                className="mb-4", style={'borderRadius': '14px', 'border': 'none'}
                            ),
                            dbc.Card(
                                dbc.CardBody([
                            html.H5(
                                "Technical notes", id="technical-notes",
                                className="mb-3",
                                style={'textDecoration': 'underline', 'fontSize': '26px'}
                            ),
                            html.H6(
                                "Shortcomings to petition data",
                                id="shortcomings", className="mb-3",
                                style={'textDecoration': 'underline', 'fontSize': '19px'}
                            ),
                            html.P(
                                "As noted above, petition data can act as a useful temperature "
                                "check, but it does have a few shortcomings that should be taken "
                                "into consideration when using the dashboard:"
                            ),
                            html.Ul([
                                html.Li([
                                    "There is currently no mechanism to stop one person signing "
                                    "the same petition twice, which means the ",
                                    html.Strong("number of signatures may be inflated"),
                                    ".",
                                    html.Sup(
                                        html.A("1", href="#about-footnote-1",
                                               style={'textDecoration': 'none'})
                                    ),
                                ]),
                                html.Li(
                                    "There is also no mechanism to check the postcode someone "
                                    "enters when they sign a petition, which means there may be "
                                    "some inaccuracies in constituency level data."
                                ),
                                html.Li([
                                    "As these petitions are hosted online, they are likely to be ",
                                    html.Strong("underrepresenting views of older constituents"),
                                    ".",
                                ]),
                            ]),
                            html.H6(
                                "Figures may differ slightly to those on the official petition "
                                "website", id="figures-differ",
                                className="mb-3 mt-5",
                                style={'textDecoration': 'underline', 'fontSize': '19px'}
                            ),
                            html.P([
                                "This website is updated automatically every morning, usually "
                                "around 4am. There may, however, be some days where it updates "
                                "later than that if there are technical issues with GitHub "
                                "Actions. Issues with GitHub Actions are recorded on: ",
                                html.A(
                                    "https://www.githubstatus.com/history",
                                    href="https://www.githubstatus.com/history",
                                    target="_blank",
                                ),
                                ".",
                            ]),
                            html.H6(
                                "Reasons for using electorate instead of population as base "
                                "size for proportions",
                                id="electorate", className="mb-1 mt-5",
                                style={'textDecoration': 'underline', 'fontSize': '19px'}
                            ),
                            html.P(
                                html.Strong(
                                    "Population estimates for Westminster Parliamentary "
                                    "constituencies in England, Wales, Northern Ireland and "
                                    "Scotland are not always comparable to each other in a way "
                                    "that electorate size is."
                                ),
                                className="mb-3"
                            ),
                            html.P(
                                "Population estimates are based on Census data, which is "
                                "collected every 10 years (most recently in 2021/22), but "
                                "Westminster constituency boundaries are reviewed every 8 years "
                                "(most recently modified in 2024). This means population "
                                "estimates for Westminster parliamentary constituencies have to "
                                "be re-aggregated as and when constituency boundaries are "
                                "redefined."
                            ),
                            html.P([
                                "Responsibility for those population estimates lie with three "
                                "different public bodies: the ONS for England and Wales, NISRA "
                                "for Northern Ireland and the NRS for Scotland. While these "
                                "three bodies work together to produce the Census,",
                                html.Sup(
                                    html.A("2", href="#about-footnote-2",
                                           style={'textDecoration': 'none'})
                                ),
                                " they do not necessarily produce population estimates for "
                                "revised Westminster constituency boundaries using the same "
                                "methodology and they may publish them at different times.",
                                html.Sup(
                                    html.A("3", href="#about-footnote-3",
                                           style={'textDecoration': 'none'})
                                ),
                                " For example, NRS has not yet produced population estimates "
                                "for the latest constituency boundaries using Census data, "
                                "although it has produced them using data from '2011 data "
                                "zones'.",
                                html.Sup(
                                    html.A("4", href="#about-footnote-4",
                                           style={'textDecoration': 'none'})
                                ),
                                " Additionally, the most recent Census was run a year later in "
                                "Scotland than in England, Wales and Northern Ireland, due to "
                                "the pandemic, creating additional complications.",
                                html.Sup(
                                    html.A("5", href="#about-footnote-5",
                                           style={'textDecoration': 'none'})
                                ),
                            ]),
                            html.P(
                                "By contrast, data on electorate size is always available for "
                                "the latest Westminster parliamentary constituency boundaries. "
                                "They are always collected at the same timepoint across all "
                                "four countries. This makes it possible to produce comparable "
                                "statistics for all 650 constituencies when using electorate "
                                "size."
                            ),
                            html.P(
                                "One shortcoming of using electorate size, however, is that it "
                                "does not necessarily represent the entire population that is "
                                "eligible to sign the petition: there is no requirement to be "
                                "registered to vote and no restriction on age or citizenship "
                                "status to sign an e-petition."
                            ),
                                ]),
                                className="mb-4", style={'borderRadius': '14px', 'border': 'none'}
                            ),
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5("Contact", id="contact", className="mb-3", style={'textDecoration': 'underline', 'fontSize': '26px'}),
                                    html.P([
                                        "For any feedback, comments or suggestions, please email ",
                                        html.A(
                                            "ukpetitionanalytics@gmail.com",
                                            href="mailto:ukpetitionanalytics@gmail.com",
                                        ),
                                        ". I aim to respond within 2-3 working days.",
                                    ]),
                                ]),
                                className="mb-4", style={'borderRadius': '14px', 'border': 'none'}
                            ),
                            dbc.Card(
                                dbc.CardBody([
                                    html.H5("Licence", id="licence", className="mb-3", style={'textDecoration': 'underline', 'fontSize': '26px'}),
                                    html.P([
                                        "This website contains public sector information licensed under the ",
                                        html.A(
                                            "Open Government Licence v3.0",
                                            href="https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
                                            target="_blank",
                                        ),
                                        " and Parliamentary information licensed the ",
                                        html.A(
                                            "Open Parliament Licence v3.0",
                                            href="https://www.parliament.uk/site-information/copyright-parliament/open-parliament-licence/",
                                            target="_blank",
                                        ),
                                        ".",
                                    ]),
                                ]),
                                className="mb-4", style={'borderRadius': '14px', 'border': 'none'}
                            ),
                            dbc.Card(
                                dbc.CardBody([
                            html.Hr(className="mt-0"),
                            html.P([
                                html.Sup("1", id="about-footnote-1"),
                                " Tasneem Ghazi, 'Are Online Petitions Useful?', ",
                                html.I("The Constitution Society"),
                                ", 29 January 2025 <",
                                html.A(
                                    "https://consoc.org.uk/blog-are-online-petitions-useful/",
                                    href="https://consoc.org.uk/blog-are-online-petitions-useful/",
                                    target="_blank",
                                ),
                                "> [accessed 1 September 2026]",
                            ], style={'fontSize': '13px'}),
                            html.P([
                                html.Sup("2", id="about-footnote-2"),
                                " Office for National Statistics, 'Combining and Comparing "
                                "Census Figures across the UK', ",
                                html.I("Census 2021"),
                                " <",
                                html.A(
                                    "www.ons.gov.uk/news/news/combiningandcomparingcensusfiguresacrosstheuk",
                                    href="https://www.ons.gov.uk/news/news/combiningandcomparingcensusfiguresacrosstheuk",
                                    target="_blank",
                                ),
                                "> [accessed 1 September 2026].",
                            ], style={'fontSize': '13px'}),
                            html.P([
                                html.Sup("3", id="about-footnote-3"),
                                " Email from Office for National Statistics to UK Petition "
                                "Analytics Owner, 4 August 2026.",
                            ], style={'fontSize': '13px'}),
                            html.P([
                                html.Sup("4", id="about-footnote-4"),
                                " According to the UK Data Service website, population "
                                "estimates for constituencies in Scotland based on the latest "
                                "boundaries and 2022 census data is 'Pending': UK Data Service, "
                                "'Scotland's Census 2022 – UV101a: Usual Resident "
                                "Population by Sex by Age (20 Categories)' <",
                                html.A(
                                    "statistics.ukdataservice.ac.uk/dataset/scotland-s-census-2022-uv101a-usual-resident-population-by-sex-by-age-20",
                                    href="https://statistics.ukdataservice.ac.uk/dataset/scotland-s-census-2022-uv101a-usual-resident-population-by-sex-by-age-20",
                                    target="_blank",
                                ),
                                "> [accessed 1 September 2026]. For population estimates based "
                                "on data zones, see: National Records of Scotland, 'Other "
                                "Geographies: Mid-2022 to Mid-2024 (2011 Data Zones)' <",
                                html.A(
                                    "https://www.nrscotland.gov.uk/publications/other-geographies-mid-2022-to-mid-2024-2011-data-zones/",
                                    href="https://www.nrscotland.gov.uk/publications/other-geographies-mid-2022-to-mid-2024-2011-data-zones/",
                                    target="_blank",
                                ),
                                "> [accessed 1 September 2026].",
                            ], style={'fontSize': '13px'}),
                            html.P([
                                html.Sup("5", id="about-footnote-5"),
                                " ONS, 'Combining and Comparing Census Figures across the UK'.",
                            ], style={'fontSize': '13px'}),
                                ]),
                                className="mb-4",
                                style={'borderRadius': '14px', 'border': 'none', 'marginTop': '40px'}
                            ),
                        ], width=8)
                    ])

                ], id='about-page-content', style={'padding': '20px'})
            ])

        ])
    ], fluid=True)
])


#####################
#### Callbacks   ####
#####################

# ── Top nav (drives the hidden dcc.Tabs) ──────────────────
#
# Each tab gets its own URL (/, /petition-overview, /all-open-petitions, /about).
# Clicking a nav link updates the URL (navlink_to_url); the URL is the single
# source of truth for which tab/navlink is active (switch_tab), so a direct
# link, a page refresh, or the browser back/forward buttons all land on the
# right tab too. Dash's default catch-all route already serves the app's
# index page for any path, so a direct request/refresh doesn't 404.

TAB_PATHS = {
    'tab-1-navlink': '/constituency-overview',
    'tab-2-navlink': '/petition-overview',
    'tab-3-navlink': '/all-open-petitions',
    'tab-4-navlink': '/about',
}


@app.callback(
    Output('url', 'pathname'),
    Input('tab-1-navlink', 'n_clicks'),
    Input('tab-2-navlink', 'n_clicks'),
    Input('tab-3-navlink', 'n_clicks'),
    Input('tab-4-navlink', 'n_clicks'),
    prevent_initial_call=True
)
def navlink_to_url(_n1, _n2, _n3, _n4):
    return TAB_PATHS.get(ctx.triggered_id, '/')


@app.callback(
    Output('main-tabs', 'value'),
    Output('tab-1-navlink', 'active'),
    Output('tab-2-navlink', 'active'),
    Output('tab-3-navlink', 'active'),
    Output('tab-4-navlink', 'active'),
    Input('url', 'pathname'),
)
def switch_tab(pathname):
    if pathname == '/petition-overview':
        return 'tab-2', False, True, False, False
    if pathname == '/all-open-petitions':
        return 'tab-3', False, False, True, False
    if pathname == '/about':
        return 'tab-4', False, False, False, True
    return 'tab-1', True, False, False, False


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
        return top5_percent_title_text("Petitions where your constituency ranks in top 5% of constituencies based on signature rate"), html.Div(
            NO_CONSTITUENCY_MESSAGE,
            style={
                'padding': '10px 20px', 'color': '#C0392B', 'fontSize': '16px', 'textAlign': 'center',
                'minHeight': '250px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'
            }
        )

    constituency_name = pcon24cds.loc[pcon24cds['PCON24CD'] == PCON24CD, 'constituency_name'].iloc[0]
    title = top5_percent_title_text(f"Petitions where {constituency_name} ranks in top 5% of constituencies based on signature rate")

    df = petitions_df[
        (petitions_df['PCON24CD'] == PCON24CD) &
        (petitions_df['status'] == 'open') &
        (petitions_df['percentile_rank_electorate'] <= 5)
    ][['petition_title', 'petition_url', 'sig_prop_electorate_rank', 'percentile_rank_electorate',
       'sig_prop_electorate', 'signature_count']] \
        .drop_duplicates(subset='petition_title') \
        .sort_values('percentile_rank_electorate', ascending=True) \
        .copy()

    if df.empty:
        return title, html.Div(
            f"No petitions currently have this constituency in the top 5% based on {SIGNATURE_RATE_LABEL.lower()}.",
            style={'padding': '10px 20px', 'color': '#777', 'fontSize': '13px'}
        )

    df['petition_title_link'] = df.apply(
        lambda r: f"[{r['petition_title']}]({r['petition_url']})", axis=1
    )
    df['sig_prop_electorate_rank'] = df['sig_prop_electorate_rank'].astype(int)
    df['sig_ratio_display'] = df.apply(
        lambda r: f"{r['sig_prop_electorate']:.2f}%\n\n({r['signature_count']:,} signatures)", axis=1
    )
    df['rank_display'] = df.apply(
        lambda r: f"{r['sig_prop_electorate_rank']} of {TOTAL_CONSTITUENCIES}  \nconstituencies", axis=1
    )

    table = dag.AgGrid(
        id='top5-percent-datatable',
        rowData=df.to_dict('records'),
        columnDefs=[
            {'field': 'petition_title_link', 'headerName': 'Petition', 'cellRenderer': 'markdown',
             'cellClass': 'petition-title-cell',
             'flex': 1.5, 'minWidth': 190, 'wrapText': True, 'autoHeight': True},
            {'field': 'rank_display', 'headerName': 'Ranking', 'cellRenderer': 'markdown',
             'cellClass': 'rank-ratio-cell', 'headerClass': 'ag-header-center',
             'flex': 1.1, 'minWidth': 130, 'wrapText': True, 'autoHeight': True},
            {'field': 'sig_ratio_display', 'headerName': SIGNATURE_RATE_LABEL, 'cellRenderer': 'markdown',
             'cellClass': 'sig-ratio-cell', 'headerClass': 'ag-header-center',
             'flex': 1.3, 'minWidth': 120, 'wrapText': True, 'autoHeight': True},
        ],
        defaultColDef={'sortable': False, 'resizable': False, 'wrapHeaderText': True, 'autoHeaderHeight': True},
        dashGridOptions={
            'domLayout': 'autoHeight',
            'animateRows': False,
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
    Output('debate-sig-prop-electorate-box', 'children'),
    Output('debate-ranking-box', 'children'),
    Output('upcoming-debates-histogram', 'figure'),
    Output('upcoming-debates-histogram', 'config'),
    Output('view-petition-link-btn', 'href'),
    Output('view-petition-link-btn', 'style'),
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
        return (
            "", "", "", blank_fig, {**histogram_config, 'staticPlot': True},
            '#', {**VIEW_PETITION_BTN_STYLE, 'display': 'none'}
        )

    df = petitions_df[petitions_df['petition_id'] == petition_id]

    total_votes = df['total_signature_count'].iloc[0]
    petition_url = df['petition_url'].iloc[0]

    constituency_row = df.loc[df['PCON24CD'] == PCON24CD] if PCON24CD is not None else None

    constituency_votes = constituency_row['signature_count'].iloc[0] if constituency_row is not None else None
    sig_prop_electorate = constituency_row['sig_prop_electorate'].iloc[0] if constituency_row is not None else None
    sig_rank = constituency_row['sig_prop_electorate_rank'].iloc[0] if constituency_row is not None else None
    sig_percentile = constituency_row['percentile_rank_electorate'].iloc[0] if constituency_row is not None else None

    constituency_name = (
        pcon24cds.loc[pcon24cds['PCON24CD'] == PCON24CD, 'constituency_name'].iloc[0]
        if PCON24CD is not None else None
    )

    fig = render_signature_histogram(
        df, df['sig_prop_electorate'].median(), sig_prop_electorate, constituency_name,
        value_col='sig_prop_electorate', x_axis_title=SIGNATURE_RATE_LABEL,
        bin_unit_label='% of electorate', value_fmt=lambda v: f"{v:.2f}%", discrete=False,
        tick_format='.2f', tick_suffix='%', hide_zero_tick=True
    )

    paren_style = {'display': 'block', 'marginTop': '6px', 'fontSize': '14px', 'fontWeight': 'normal', 'color': '#888'}

    total_votes_str = f"{total_votes:,}"
    sig_prop_electorate_str = (
        html.Span([
            f"{sig_prop_electorate:.2f}%",
            html.Span(f"({constituency_votes:,} signatures)", style=paren_style)
        ])
        if sig_prop_electorate is not None else no_constituency
    )
    category = percentile_category(sig_percentile)
    ranking_str = (
        html.Span([f"{int(sig_rank)} of {TOTAL_CONSTITUENCIES}"] + ([html.Span(f"({category})", style=paren_style)] if category else []))
        if pd.notna(sig_rank) else (no_constituency if PCON24CD is None else no_ranking)
    )

    return (
        total_votes_str, sig_prop_electorate_str, ranking_str, fig, histogram_config,
        petition_url, {**VIEW_PETITION_BTN_STYLE, 'display': 'inline-block'}
    )


# ── All Petitions table ───────────────────────────────────
#
# Patches the already-mounted 'all-petitions-datatable' grid's rowData/columnDefs
# directly (see build_all_petitions_table() above) instead of replacing the whole
# component on every constituency change — see the comment above
# _petitions_display_base for why. Runs on initial load too (no prevent_initial_call)
# because the constituency dropdown's persisted value may already select a
# constituency by the time the page first renders, and this grid needs to reflect
# that rather than sitting on the layout's no-constituency-selected placeholder.

@app.callback(
    Output('all-petitions-datatable', 'rowData'),
    Output('all-petitions-datatable', 'columnDefs'),
    Input('analytics-petition-dropdown', 'value'),
)
def update_all_petitions_table(PCON24CD):
    return _build_all_petitions_rowdata(PCON24CD), _build_all_petitions_columndefs(PCON24CD)


# ── Map tab ───────────────────────────────────────────────

@app.callback(
    Output('petition-histogram', 'figure'),
    Output('total-sigs', 'children'),
    Output('date-opened', 'children'),
    Output('sch-debate-date', 'children'),
    Output('petition-median-sig-rate-box', 'children'),
    Output('petition-sig-prop-electorate-box', 'children'),
    Output('petition-sig-prop-electorate-rank-box', 'children'),
    Output('petition-top-constituencies-table', 'children'),
    Output('view-petition-link-btn-2', 'href'),
    Output('view-petition-link-btn-2', 'style'),
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

    petition_url = df['petition_url'].iloc[0]

    opened_at = df['opened_at'].iloc[0] if 'opened_at' in df.columns else None
    opened_at_str = str(opened_at) if pd.notna(opened_at) else ""

    sch_debate_date = df['scheduled_debate_date'].iloc[0] if 'scheduled_debate_date' in df.columns else None
    debate_date_str = str(sch_debate_date) if pd.notna(sch_debate_date) else "Not scheduled"

    constituency_row = df.loc[df['PCON24CD'] == PCON24CD] if PCON24CD is not None else None
    constituency_name = (
        pcon24cds.loc[pcon24cds['PCON24CD'] == PCON24CD, 'constituency_name'].iloc[0]
        if PCON24CD is not None else None
    )

    sig_prop_electorate = constituency_row['sig_prop_electorate'].iloc[0] if constituency_row is not None else None
    constituency_signature_count = constituency_row['signature_count'].iloc[0] if constituency_row is not None else None
    sig_prop_electorate_rank = constituency_row['sig_prop_electorate_rank'].iloc[0] if constituency_row is not None else None
    sig_prop_electorate_percentile = constituency_row['percentile_rank_electorate'].iloc[0] if constituency_row is not None else None

    median_sig_rate = df['sig_prop_electorate'].median()
    sorted_by_rate = df.sort_values('sig_prop_electorate').reset_index(drop=True)
    mid = len(sorted_by_rate) // 2
    if len(sorted_by_rate) % 2 == 0:
        median_signature_count = round((sorted_by_rate.loc[mid - 1, 'signature_count'] + sorted_by_rate.loc[mid, 'signature_count']) / 2)
    else:
        median_signature_count = sorted_by_rate.loc[mid, 'signature_count']

    histogram_fig = render_signature_histogram(
        df, median_sig_rate, sig_prop_electorate, constituency_name,
        value_col='sig_prop_electorate', x_axis_title=SIGNATURE_RATE_LABEL,
        bin_unit_label='% of electorate', value_fmt=lambda v: f"{v:.3f}%", discrete=False,
        tick_format='.3f', tick_suffix='%', hide_zero_tick=True
    )

    median_sig_rate_str = html.Span([
        f"{median_sig_rate:.2f}%",
        html.Span(f"({median_signature_count:,.0f} signatures)", style=paren_style)
    ])
    sig_prop_electorate_str = (
        html.Span([
            f"{sig_prop_electorate:.2f}%",
            html.Span(f"({constituency_signature_count:,.0f} signatures)", style=paren_style)
        ])
        if sig_prop_electorate is not None else no_constituency
    )
    electorate_category = percentile_category(sig_prop_electorate_percentile)
    sig_prop_electorate_rank_str = (
        html.Span([f"{int(sig_prop_electorate_rank)} of {TOTAL_CONSTITUENCIES}"] + ([html.Span(f"({electorate_category})", style=paren_style)] if electorate_category else []))
        if pd.notna(sig_prop_electorate_rank) else (no_constituency if PCON24CD is None else no_ranking)
    )

    top_constituencies = df[['constituency_name', 'sig_prop_electorate', 'signature_count']] \
        .sort_values('sig_prop_electorate', ascending=False) \
        .reset_index(drop=True)

    sig_prop_electorate_uf = calculate_upperfence(top_constituencies['sig_prop_electorate'].dropna())

    top_constituencies_table = dag.AgGrid(
        id='petition-top-constituencies-datatable',
        rowData=top_constituencies.to_dict('records'),
        columnDefs=[
            {'field': 'constituency_name', 'headerName': 'Constituency', 'flex': 2, 'minWidth': 180},
            {'field': 'sig_prop_electorate', 'headerName': 'Signature rate',
             'valueFormatter': {'function': "params.value == null ? '' : params.value.toFixed(3) + '%'"},
             'flex': 1, 'minWidth': 110, 'cellStyle': {'textAlign': 'center'}, 'headerClass': 'ag-header-center',
             'sort': 'desc'},
            {'field': 'signature_count', 'headerName': 'Number of signatures',
             'valueFormatter': {'function': "params.value == null ? '' : params.value.toLocaleString()"},
             'flex': 1, 'minWidth': 110, 'cellStyle': {'textAlign': 'center'}, 'headerClass': 'ag-header-center'},
        ],
        defaultColDef={'sortable': True, 'resizable': False, 'wrapHeaderText': True, 'autoHeaderHeight': True},
        dashGridOptions={'unSortIcon': True},
        getRowStyle={'styleConditions': [
            {'condition': f'params.data.sig_prop_electorate >= {sig_prop_electorate_uf}',
             'style': {'backgroundColor': '#E6D9F2'}}
        ]},
        className='ag-theme-alpine',
        style={'width': '100%', 'height': '100%'},
    )

    return (
        histogram_fig,
        f"{total_signatures:,}",
        opened_at_str,
        debate_date_str,
        median_sig_rate_str,
        sig_prop_electorate_str,
        sig_prop_electorate_rank_str,
        top_constituencies_table,
        petition_url, {**VIEW_PETITION_BTN_STYLE, 'display': 'inline-block'}
    )


print(f"[startup] Total time until app is ready to serve: {time.time() - _startup_t0:.2f}s")

if __name__ == '__main__':
    app.run(debug=(ENV == 'local'), port=8051)