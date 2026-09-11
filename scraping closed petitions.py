"""
1) Re-scrapes the petitions that closed yesterday (read from yesterday's
   petitions_closed_{date} file, if it exists) to capture their final state -
   opened/deadline/debate dates and per-constituency signature counts - since once
   a petition closes it drops out of "daily web scraping open petitions.py"'s
   state=open scrape and would otherwise never be captured again.
2) Filters today's open-petitions list down to petitions whose deadline is today
   and writes today's petitions_closed_{date} file, which becomes the input for
   step 1 tomorrow.
"""

import boto3
import gzip
import pandas as pd
from datetime import date, timedelta, datetime
import os
import time
import aiohttp
import asyncio
import ssl
import certifi
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv

script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

ENV = os.getenv('ENV', 'production')

today = date.today()
today_str = today.strftime('%Y%m%d')
yesterday_str = (today - timedelta(days=1)).strftime('%Y%m%d')

bucket = 'uk-petitions-dashboard'

headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"}

SEMAPHORE = None
fetch_counter = 0


# ── S3 / local helpers ────────────────────────────────────

def load_csv_from_s3(s3_client, filename):
    s3_object = s3_client.get_object(Bucket=bucket, Key=filename)
    body = s3_object['Body'].read()
    if filename.endswith('.gz'):
        body = gzip.decompress(body)
    return pd.read_csv(BytesIO(body))


def load_dynamic_csv(s3_client, base_key):
    """Prefers the gzip-compressed key (base_key + '.gz'), falling back to the
    legacy uncompressed key - mirrors dashboard.py's load_dynamic_csv()."""
    try:
        return load_csv_from_s3(s3_client, f'{base_key}.gz')
    except s3_client.exceptions.NoSuchKey:
        return load_csv_from_s3(s3_client, base_key)


def upload_to_s3(df, file_name, s3_client):
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    compressed = gzip.compress(csv_bytes)
    s3_client.put_object(
        Bucket=bucket,
        Key=file_name,
        Body=compressed,
        ContentEncoding='gzip',
        ContentType='text/csv'
    )


# ── Petition detail scraping (same fields as "daily web scraping open petitions.py") ──

async def fetch_json(session, url, max_retries=5):
    global fetch_counter
    for attempt in range(max_retries):
        async with SEMAPHORE:
            async with session.get(f"{url}.json", headers=headers) as response:
                if response.status == 429:
                    wait = 3 ** attempt
                    print(f"Rate limited on {url}, retrying in {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                elif response.status == 200:
                    fetch_counter += 1
                    if fetch_counter % 100 == 0:
                        print(f"Fetched {fetch_counter} requests so far...")
                    await asyncio.sleep(1.0)
                    return await response.json()
                else:
                    print(f"Failed {url}: status {response.status}")
                    return None
    print(f"Giving up on {url} after {max_retries} attempts")
    return None


async def fetch_petition_full(session, url, petition_id):
    data = await fetch_json(session, url)
    if not data:
        return None, None, None, None, None, None, None, None, []

    attrs = data['data']['attributes']

    opened = None
    deadline = None
    closed_at = None
    debate_threshold_reached_date = None
    response_threshold_reached_date = None
    scheduled_debate_date = None
    if attrs.get('opened_at'):
        opened = datetime.strptime(attrs['opened_at'], '%Y-%m-%dT%H:%M:%S.%fZ').date()
    # Once a petition closes, the API stops reporting 'closing_date' (it goes null) and
    # moves the closure info to 'closed_at' instead - fall back to that so a closed
    # petition doesn't lose its deadline on re-scrape.
    if attrs.get('closing_date'):
        deadline = datetime.strptime(attrs['closing_date'], '%Y-%m-%d').date()
    elif attrs.get('closed_at'):
        deadline = datetime.strptime(attrs['closed_at'], '%Y-%m-%dT%H:%M:%S.%fZ').date()
    if attrs.get('closed_at'):
        closed_at = datetime.strptime(attrs['closed_at'], '%Y-%m-%dT%H:%M:%S.%fZ').date()
    if attrs.get('debate_threshold_reached_at'):
        debate_threshold_reached_date = datetime.strptime(attrs['debate_threshold_reached_at'], '%Y-%m-%dT%H:%M:%S.%fZ').date()
    if attrs.get('response_threshold_reached_at'):
        response_threshold_reached_date = datetime.strptime(attrs['response_threshold_reached_at'], '%Y-%m-%dT%H:%M:%S.%fZ').date()
    if attrs.get('scheduled_debate_date'):
        scheduled_debate_date = datetime.strptime(attrs['scheduled_debate_date'], '%Y-%m-%d').date()

    state = attrs.get('state')
    total_signature_count = attrs.get('signature_count')

    constituency_records = [
        {
            'petition_id': petition_id,
            'PCON24CD': c['ons_code'],
            'constituency_name': c['name'],
            'signature_count': c['signature_count']
        }
        for c in attrs['signatures_by_constituency']
    ]

    return opened, deadline, closed_at, debate_threshold_reached_date, response_threshold_reached_date, scheduled_debate_date, state, total_signature_count, constituency_records


async def scrape_petitions(df):
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(limit=10, limit_per_host=5, ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            fetch_petition_full(session, petition['petition_url'], petition['petition_id'])
            for _, petition in df.iterrows()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return results


# ── Step 1: re-scrape petitions that closed yesterday ─────

def load_yesterdays_closed_petitions(s3_client):
    """Returns the petitions_closed_{yesterday} dataframe, or None if it doesn't exist."""
    if ENV == 'local':
        path = script_dir / 'cached_data' / f'petitions_closed_{yesterday_str}.csv'
        if not path.exists():
            return None
        return pd.read_csv(path)
    else:
        try:
            return load_csv_from_s3(s3_client, f'holding_data/petitions_closed_{yesterday_str}.csv.gz')
        except s3_client.exceptions.NoSuchKey:
            return None


async def rescrape_yesterdays_closed_petitions(closed_yesterday, s3_client):
    print(f"1. Re-scraping {len(closed_yesterday)} petition(s) that closed yesterday ({yesterday_str})...")
    start_time = time.time()
    results = await scrape_petitions(closed_yesterday)
    print(f"   Time taken: {time.time() - start_time:.2f} seconds")

    opened_list, deadline_list, closed_at_list, debate_threshold_list = [], [], [], []
    response_threshold_list, scheduled_debate_list = [], []
    status_list, total_signature_count_list, counts_lists = [], [], []
    for r in results:
        if isinstance(r, Exception):
            opened_list.append(None)
            deadline_list.append(None)
            closed_at_list.append(None)
            debate_threshold_list.append(None)
            response_threshold_list.append(None)
            scheduled_debate_list.append(None)
            status_list.append(None)
            total_signature_count_list.append(None)
            counts_lists.append([])
        else:
            opened, deadline, closed_at, debate_threshold, response_threshold, scheduled_debate, status, total_signature_count, records = r
            opened_list.append(opened)
            deadline_list.append(deadline)
            closed_at_list.append(closed_at)
            debate_threshold_list.append(debate_threshold)
            response_threshold_list.append(response_threshold)
            scheduled_debate_list.append(scheduled_debate)
            status_list.append(status)
            total_signature_count_list.append(total_signature_count)
            counts_lists.append(records)

    closed_yesterday = closed_yesterday.copy()
    closed_yesterday['opened_at'] = opened_list
    # A closed petition's deadline is fixed and already known from when it was first
    # captured (while still open) - only overwrite it if the re-scrape found a value,
    # rather than losing it if a fetch failed.
    closed_yesterday['deadline'] = pd.Series(deadline_list, index=closed_yesterday.index) \
        .combine_first(closed_yesterday['deadline'])
    closed_yesterday['closed_at'] = closed_at_list
    closed_yesterday['debate_threshold_reached_at'] = debate_threshold_list
    closed_yesterday['response_threshold_reached_at'] = response_threshold_list
    closed_yesterday['scheduled_debate_date'] = scheduled_debate_list
    closed_yesterday['status'] = pd.Series(status_list, index=closed_yesterday.index) \
        .combine_first(closed_yesterday['status'])
    closed_yesterday['total_signature_count'] = pd.Series(total_signature_count_list, index=closed_yesterday.index) \
        .combine_first(closed_yesterday['total_signature_count'])

    closed_petition_counts_df = pd.DataFrame([item for sublist in counts_lists for item in sublist])

    print(f"   Total petitions: {len(closed_yesterday)}")
    print(f"   Total constituency records: {len(closed_petition_counts_df)}")

    if ENV == 'local':
        cache_dir = script_dir / 'cached_data'
        cache_dir.mkdir(exist_ok=True)
        closed_yesterday.to_csv(cache_dir / f'closed_petitions_list_{today_str}.csv', index=False)
        closed_petition_counts_df.to_csv(cache_dir / f'closed_petitions_counts_{today_str}.csv', index=False)
        print(f"   Saved to {cache_dir}")
    else:
        upload_to_s3(closed_yesterday, f'dynamic_data/closed_petitions_list_{today_str}.csv.gz', s3_client)
        upload_to_s3(closed_petition_counts_df, f'dynamic_data/closed_petitions_counts_{today_str}.csv.gz', s3_client)
        print("   Upload complete!")


# ── Step 2: build today's petitions_closed file ────────────

def build_todays_closed_file(s3_client):
    print(f"2. Finding petitions closing today...")

    if ENV == 'local':
        list_path = script_dir / 'cached_data' / f'petitions_list_{today_str}.csv'
        petitions_list = pd.read_csv(list_path)
    else:
        petitions_list = load_dynamic_csv(s3_client, f'dynamic_data/petitions_list_{today_str}.csv')

    closing_today = petitions_list[petitions_list['deadline'] == today.isoformat()].copy()

    print(f"   {len(closing_today)} petition(s) closing today ({today.isoformat()}):")
    for _, row in closing_today.iterrows():
        print(f"   - [{row['petition_id']}] {row['petition_title']} ({row['total_signature_count']} signatures)")

    if closing_today.empty:
        print("   No petitions closing today - skipping file write.")
    elif ENV == 'local':
        out_path = script_dir / 'cached_data' / f'petitions_closed_{today_str}.csv'
        closing_today.to_csv(out_path, index=False)
        print(f"   Saved to {out_path}")
    else:
        upload_to_s3(closing_today, f'holding_data/petitions_closed_{today_str}.csv.gz', s3_client)
        print("   Upload complete!")


async def main():
    global SEMAPHORE
    # No time pressure here - typically only a handful of petitions closed yesterday,
    # so keep concurrency low and pace requests gently rather than racing like the
    # full open-petitions scrape does.
    SEMAPHORE = asyncio.Semaphore(3)

    s3_client = None
    if ENV != 'local':
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )

    closed_yesterday = load_yesterdays_closed_petitions(s3_client)
    if closed_yesterday is None:
        print(f"1. No petitions_closed file found for yesterday ({yesterday_str}) - skipping re-scrape.")
    elif closed_yesterday.empty:
        print(f"1. Yesterday's ({yesterday_str}) petitions_closed file is empty - nothing to re-scrape.")
    else:
        await rescrape_yesterdays_closed_petitions(closed_yesterday, s3_client)

    build_todays_closed_file(s3_client)

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
