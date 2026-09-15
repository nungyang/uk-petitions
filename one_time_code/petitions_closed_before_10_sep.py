"""
One-time backfill: scrapes the full historical closed-petitions list from
https://petition.parliament.uk/petitions.csv?state=closed and captures each
petition's final state - opened/deadline/closed/debate dates and
per-constituency signature counts - the same fields "scraping closed
petitions.py" captures for petitions that close going forward. Needed
because that script only re-scrapes petitions that closed *yesterday*, so
everything that closed before it started running would otherwise be missing
opened_at/closed_at/debate dates and constituency breakdowns.

Runs slowly and at low concurrency on purpose - getting rate limited and
silently dropping petitions is worse than the run taking a while. After
scraping, rows for petitions that closed on or before 9 September 2026 are
dropped (from both the list and the counts), since only the 10 September
2026 onward gap needs backfilling here. Uploads the result to S3 under
'closed_petitions/'.
"""

import pandas as pd
import requests
import boto3
import gzip
from datetime import datetime, date
import time
import aiohttp
import asyncio
import ssl
import certifi
import os
from pathlib import Path
from io import StringIO
from dotenv import load_dotenv

repo_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=repo_root / '.env')

LIMIT = None  # set to an int to test with a subset
CUTOFF_DATE = date(2026, 9, 10)  # keep only petitions that closed on/after this date
bucket = 'uk-petitions-dashboard'

headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"}

SEMAPHORE = None
fetch_counter = 0


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


async def fetch_json(session, url, max_retries=8):
    global fetch_counter
    for attempt in range(max_retries):
        async with SEMAPHORE:
            async with session.get(f"{url}.json", headers=headers) as response:
                if response.status == 429:
                    wait = min(60, 5 * (attempt + 1))
                    print(f"Rate limited on {url}, retrying in {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                elif response.status == 200:
                    fetch_counter += 1
                    if fetch_counter % 100 == 0:
                        print(f"Fetched {fetch_counter} requests so far...")
                    await asyncio.sleep(1.5)
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
    # Closed petitions have 'closing_date' set to null by the API - fall back to
    # 'closed_at' so the deadline isn't lost.
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
    connector = aiohttp.TCPConnector(limit=5, limit_per_host=3, ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            fetch_petition_full(session, petition['petition_url'], petition['petition_id'])
            for _, petition in df.iterrows()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return results


async def main():
    global SEMAPHORE
    # Deliberately low concurrency - avoiding rate limiting matters more than speed here.
    SEMAPHORE = asyncio.Semaphore(3)

    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )

    print("1. Fetching full closed-petitions list...")
    url = 'https://petition.parliament.uk/petitions.csv?state=closed'
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    closed_petitions = pd.read_csv(StringIO(response.text))

    closed_petitions = closed_petitions.rename(columns={
        'Petition': 'petition_title',
        'URL': 'petition_url',
        'State': 'status',
        'Signatures Count': 'total_signature_count'
    })
    closed_petitions['petition_id'] = closed_petitions['petition_url'].str.split('/').str[-1]
    print(f"   Found {len(closed_petitions)} closed petitions in total")

    if LIMIT is not None:
        closed_petitions = closed_petitions.head(LIMIT).copy()
        print(f"   Test run - limiting to first {len(closed_petitions)} petitions")

    print("\n2. Fetching petition details (dates + constituency signatures)...")
    start_time = time.time()
    results = await scrape_petitions(closed_petitions)
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

    closed_petitions['opened_at'] = opened_list
    closed_petitions['deadline'] = deadline_list
    closed_petitions['closed_at'] = closed_at_list
    closed_petitions['debate_threshold_reached_at'] = debate_threshold_list
    closed_petitions['response_threshold_reached_at'] = response_threshold_list
    closed_petitions['scheduled_debate_date'] = scheduled_debate_list
    # Fall back to the values already in the closed-petitions CSV if a fetch failed.
    closed_petitions['status'] = pd.Series(status_list, index=closed_petitions.index) \
        .combine_first(closed_petitions['status'])
    closed_petitions['total_signature_count'] = pd.Series(total_signature_count_list, index=closed_petitions.index) \
        .combine_first(closed_petitions['total_signature_count'])

    closed_petition_counts_df = pd.DataFrame([item for sublist in counts_lists for item in sublist])

    print(f"\n3. Summary...")
    print(f"   Total petitions: {len(closed_petitions)}")
    print(f"   Total constituency records: {len(closed_petition_counts_df)}")

    print(f"\n4. Dropping petitions that closed before {CUTOFF_DATE}...")
    closed_at_dates = pd.to_datetime(closed_petitions['closed_at'], errors='coerce').dt.date
    # Rows with an unknown closed_at (failed fetch) carry no data either - drop them too.
    keep_mask = closed_at_dates >= CUTOFF_DATE
    dropped = len(closed_petitions) - keep_mask.sum()
    closed_petitions = closed_petitions[keep_mask].copy()
    kept_ids = set(closed_petitions['petition_id'])
    closed_petition_counts_df = closed_petition_counts_df[
        closed_petition_counts_df['petition_id'].isin(kept_ids)
    ].copy()

    print(f"   Dropped {dropped} petition(s) closed before {CUTOFF_DATE}")
    print(f"   Remaining petitions: {len(closed_petitions)}")
    print(f"   Remaining constituency records: {len(closed_petition_counts_df)}")

    print("\n5. Saving to local cache...")
    cache_dir = repo_root / 'cached_data'
    cache_dir.mkdir(exist_ok=True)

    closed_petitions.to_csv(cache_dir / 'closed_petitions_list_backfill.csv', index=False)
    closed_petition_counts_df.to_csv(cache_dir / 'closed_petitions_list_counts_backfill.csv', index=False)
    print(f"   Saved to {cache_dir}")

    print("\n6. Uploading to S3...")
    upload_to_s3(closed_petitions, 'closed_petitions/closed_petitions_list_backfill.csv.gz', s3_client)
    upload_to_s3(closed_petition_counts_df, 'closed_petitions/closed_petitions_list_counts_backfill.csv.gz', s3_client)
    print("   Upload complete!")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
