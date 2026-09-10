"""
Web scraping from UK petitions website
"""

# Importing libraries
import boto3
import gzip
import pandas as pd
import requests
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import time
import aiohttp
import asyncio
from io import StringIO
from dotenv import load_dotenv
import ssl
import certifi
from pathlib import Path

# For when I run locally
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

ENV = os.getenv('ENV', 'production')

today = date.today()

headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"}

SEMAPHORE = None
fetch_counter = 0


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
                    await asyncio.sleep(0.4)
                    return await response.json()
                else:
                    print(f"Failed {url}: status {response.status}")
                    return None
    print(f"Giving up on {url} after {max_retries} attempts")
    return None


async def fetch_petition_full(session, url, petition_id):
    data = await fetch_json(session, url)
    if not data:
        return None, None, None, None, []

    attrs = data['data']['attributes']

    opened = None
    deadline = None
    debate_threshold_reached_date = None
    scheduled_debate_date = None
    if attrs.get('opened_at'):
        opened = datetime.strptime(attrs['opened_at'], '%Y-%m-%dT%H:%M:%S.%fZ').date()
    if attrs.get('closing_date'):
        deadline = datetime.strptime(attrs['closing_date'], '%Y-%m-%d').date()
    if attrs.get('debate_threshold_reached_at'):
        debate_threshold_reached_date = datetime.strptime(attrs['debate_threshold_reached_at'], '%Y-%m-%dT%H:%M:%S.%fZ').date()
    if attrs.get('scheduled_debate_date'):
        scheduled_debate_date = datetime.strptime(attrs['scheduled_debate_date'], '%Y-%m-%d').date()

    constituency_records = [
        {
            'petition_id': petition_id,
            'PCON24CD': c['ons_code'],
            'constituency_name': c['name'],
            'signature_count': c['signature_count']
        }
        for c in attrs['signatures_by_constituency']
    ]

    return opened, deadline, debate_threshold_reached_date, scheduled_debate_date, constituency_records


async def run(df):
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


def upload_to_s3(df, file_name, s3_client, bucket):
    # Gzip-compressed on the way up (CSVs of mostly repetitive integers compress very
    # well) so the dashboard's startup download - the dominant cost in its cold-start
    # time - has far fewer bytes to pull over the wire.
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    compressed = gzip.compress(csv_bytes)

    s3_client.put_object(
        Bucket=bucket,
        Key=file_name,
        Body=compressed,
        ContentEncoding='gzip',
        ContentType='text/csv'
    )


async def main():
    global SEMAPHORE
    SEMAPHORE = asyncio.Semaphore(10)

    print("Starting UK petitions data collection...")

    # Downloading list of all open petitions
    print("\n1. Fetching open petitions...")
    url = 'https://petition.parliament.uk/petitions.csv?state=open'
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    open_petitions = pd.read_csv(StringIO(response.text))

    # Cleaning
    open_petitions = open_petitions.rename(columns={
        'Petition': 'petition_title',
        'URL': 'petition_url',
        'State': 'status',
        'Signatures Count': 'total_signature_count'
    })
    open_petitions['petition_id'] = open_petitions['petition_url'].str.split('/').str[-1]
    print(f"   Found {len(open_petitions)} open petitions")

    # Fetching dates and constituency data for open petitions (single request per petition)
    print("\n2. Fetching petition details (dates + constituency signatures)...")
    start_time = time.time()
    results = await run(open_petitions)
    print(f"   Time taken: {time.time() - start_time:.2f} seconds")

    opened_list, deadline_list, debate_threshold_list, scheduled_debate_list, counts_lists = [], [], [], [], []
    for r in results:
        if isinstance(r, Exception):
            opened_list.append(None)
            deadline_list.append(None)
            debate_threshold_list.append(None)
            scheduled_debate_list.append(None)
            counts_lists.append([])
        else:
            opened, deadline, debate_threshold, scheduled_debate, records = r
            opened_list.append(opened)
            deadline_list.append(deadline)
            debate_threshold_list.append(debate_threshold)
            scheduled_debate_list.append(scheduled_debate)
            counts_lists.append(records)

    open_petitions['opened_at'] = opened_list
    open_petitions['deadline'] = deadline_list
    open_petitions['debate_threshold_reached_at'] = debate_threshold_list
    open_petitions['scheduled_debate_date'] = scheduled_debate_list

    open_petition_counts_df = pd.DataFrame([item for sublist in counts_lists for item in sublist])

    print(f"\n3. Summary...")
    print(f"   Total petitions: {len(open_petitions)}")
    print(f"   Total constituency records: {len(open_petition_counts_df)}")

    today_str = today.strftime('%Y%m%d')

    if ENV == 'local':
        # Saving to local cache instead of S3
        print("\n4. Saving to local cache...")
        cache_dir = script_dir / 'cached_data'
        cache_dir.mkdir(exist_ok=True)

        open_petitions.to_csv(cache_dir / f'petitions_list_{today_str}.csv', index=False)
        open_petition_counts_df.to_csv(cache_dir / f'petitions_counts_{today_str}.csv', index=False)

        print(f"Saved to {cache_dir}")
    else:
        # Exporting to Amazon S3
        print("\n4. Uploading to S3...")
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_DEFAULT_REGION')
        bucket = 'uk-petitions-dashboard'

        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )

        upload_to_s3(open_petitions, f'dynamic_data/petitions_list_{today_str}.csv.gz', s3_client, bucket)
        upload_to_s3(open_petition_counts_df, f'dynamic_data/petitions_counts_{today_str}.csv.gz', s3_client, bucket)

        print("Upload complete!")

    print("\nData collection finished successfully!")


if __name__ == "__main__":
    asyncio.run(main())