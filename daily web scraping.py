#!/usr/bin/env python
# coding: utf-8

"""
Web scraping from UK petitions website
"""

# Importing libraries
from bs4 import BeautifulSoup
import boto3
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import time
import aiohttp
import asyncio
import re
from io import StringIO
from dotenv import load_dotenv
import ssl
import certifi
from pathlib import Path

# For when I run locally
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

today = date.today()

headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"}


# Function definitions
async def fetch_dates_for_closed_petitions(urls):
    async def open_and_closed_dates(session, URL):
        try:
            async with session.get(f"{URL}.json", headers=headers) as response:
                data = await response.json()
                attrs = data['data']['attributes']
                
                opened = None
                debate_threshold_reached_date = None
                scheduled_debate_date = None
                closed = None
                
                if attrs.get('opened_at'):
                    opened = datetime.strptime(attrs['opened_at'], '%Y-%m-%dT%H:%M:%S.%fZ').date()
                if attrs.get('debate_threshold_reached_at'):
                    debate_threshold_reached_date = datetime.strptime(attrs['debate_threshold_reached_at'], '%Y-%m-%dT%H:%M:%S.%fZ').date()
                if attrs.get('scheduled_debate_date'):
                    scheduled_debate_date = datetime.strptime(attrs['scheduled_debate_date'], '%Y-%m-%d').date()
                if attrs.get('closed_at'):
                    closed = datetime.strptime(attrs['closed_at'], '%Y-%m-%dT%H:%M:%S.%fZ').date()
                    
                return opened, debate_threshold_reached_date, scheduled_debate_date, closed

        except Exception as e:
            print(f"Error processing {URL}: {e}")
            return None, None, None, None
    
    async with aiohttp.ClientSession() as session:
        tasks = [open_and_closed_dates(session, url) for url in urls]
        return await asyncio.gather(*tasks)


async def fetch_dates_for_open_petitions(urls):
    async def open_and_deadline_dates(session, URL):
        opened = None
        deadline = None
        
        try:
            # Get the HTML page
            async with session.get(URL, headers=headers) as response:
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract deadline date from HTML
                deadline_li = soup.find('li', {'class': 'meta-deadline'})
                if deadline_li:
                    deadline_text = deadline_li.get_text(strip=True)
                    match = re.search(r'(\d{1,2} \w+ \d{4})', deadline_text)
                    if match:
                        deadline = datetime.strptime(match.group(1), '%d %B %Y').date()
            
            # Get the JSON data
            async with session.get(f"{URL}.json", headers=headers) as response:
                data = await response.json()
                attrs = data['data']['attributes']
                
                debate_threshold_reached_date = None
                debate_date = None
                
                if attrs.get('opened_at'):
                    opened = datetime.strptime(attrs['opened_at'], '%Y-%m-%dT%H:%M:%S.%fZ').date()
                if attrs.get('debate_threshold_reached_at'):
                    debate_threshold_reached_date = datetime.strptime(attrs['debate_threshold_reached_at'], '%Y-%m-%dT%H:%M:%S.%fZ').date()
                if attrs.get('scheduled_debate_date'):
                    debate_date = datetime.strptime(attrs['scheduled_debate_date'], '%Y-%m-%d').date()

            return opened, deadline, debate_threshold_reached_date, debate_date
            
        except Exception as e:
            print(f"Error processing {URL}: {e}")
            return None, None, None, None
    
    async with aiohttp.ClientSession() as session:
        tasks = [open_and_deadline_dates(session, url) for url in urls]
        return await asyncio.gather(*tasks)


async def fetch_petition_data(session, url, petition_id, headers):
    try:
        async with session.get(f"{url}.json", headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                return [
                    {
                        'petition_id': petition_id,
                        'PCON24CD': c['ons_code'],
                        'constituency_name': c['name'],
                        'signature_count': c['signature_count']
                    }
                    for c in data['data']['attributes']['signatures_by_constituency']
                ]
            else:
                print(f"Failed to fetch {url} with status code {response.status}")
                return []
    except Exception as e:
        print(f"Error fetching data for petition {petition_id}: {e}")
        return []


async def run(df):
    # Create SSL context using certifi's certificates
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    # Increase connection limits for faster parallel requests
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=30, ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            fetch_petition_data(session, petition['petition_url'], petition['petition_id'], headers)
            for _, petition in df.iterrows()
        ]
        # return_exceptions=True prevents one failure from stopping everything
        all_petitions_counts = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out any exceptions
        all_petitions_counts = [item for item in all_petitions_counts if not isinstance(item, Exception)]

    return pd.DataFrame([item for sublist in all_petitions_counts for item in sublist])


def upload_to_s3(df, file_name, s3_client, bucket):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    s3_client.put_object(
        Bucket=bucket,
        Key=file_name,
        Body=csv_buffer.getvalue()
    )


async def main():
    print("Starting UK petitions data collection...")
    
    # Downloading list of all closed petitions
    print("\n1. Fetching closed petitions...")
    url = 'https://petition.parliament.uk/petitions.csv?state=closed'
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    all_closed_petitions = pd.read_csv(StringIO(response.text))
    
    # Cleaning
    all_closed_petitions = all_closed_petitions.rename(columns={
        'Petition': 'petition_title',
        'URL': 'petition_url',
        'State': 'status',
        'Signatures Count': 'total_signature_count'
    })
    all_closed_petitions['petition_id'] = all_closed_petitions['petition_url'].str.split('/').str[-1]
    
    # Fetching dates for closed petitions
    print("   Fetching dates for closed petitions...")
    results = await fetch_dates_for_closed_petitions(all_closed_petitions['petition_url'])
    all_closed_petitions['opened_at'] = [r[0] for r in results]
    all_closed_petitions['debate_threshold_reached_at'] = [r[1] for r in results]
    all_closed_petitions['scheduled_debate_date'] = [r[2] for r in results]
    all_closed_petitions['closed_at'] = [r[3] for r in results]
    
    # Fetching constituency data only for those that closed today
    yesterday = datetime.now().date() - timedelta(days=1)
    recent_closed_petitions = all_closed_petitions[all_closed_petitions['closed_at'] == yesterday]
    print(f"   Found {len(recent_closed_petitions)} petitions closed today")
    
    print("   Fetching constituency data for petitions that closed today...")
    start_time = time.time()
    closed_petition_counts_df = await run(recent_closed_petitions)
    print(f"   Time taken: {time.time() - start_time:.2f} seconds")
    
    # Downloading list of all open petitions
    print("\n2. Fetching open petitions...")
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
    
    # Fetching dates for open petitions
    print("   Fetching dates for open petitions...")
    results = await fetch_dates_for_open_petitions(open_petitions['petition_url'])
    open_petitions['opened_at'] = [r[0] for r in results]
    open_petitions['deadline'] = [r[1] for r in results]
    open_petitions['debate_threshold_reached_at'] = [r[2] for r in results]
    open_petitions['scheduled_debate_date'] = [r[3] for r in results]
    print(f"   Found {len(open_petitions)} open petitions")
    
    # Fetching constituency data for open petitions
    print("   Fetching constituency data for open petitions...")
    start_time = time.time()
    open_petition_counts_df = await run(open_petitions)
    print(f"   Time taken: {time.time() - start_time:.2f} seconds")
    
    # Merging all datasets together
    print("\n3. Merging datasets...")
    all_petitions_list = pd.concat([all_closed_petitions, open_petitions], ignore_index=True)
    all_petitions_counts = pd.concat([closed_petition_counts_df, open_petition_counts_df], ignore_index=True)
    
    print(f"   Total petitions: {len(all_petitions_list)}")
    print(f"   Total constituency records: {len(all_petitions_counts)}")
    
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
    
    today_str = today.strftime('%Y%m%d')

    upload_to_s3(all_petitions_list, f'petitions_list_{today_str}.csv', s3_client, bucket)
    upload_to_s3(all_petitions_counts, f'petitions_counts_{today_str}.csv', s3_client, bucket)
    
    print("Upload complete!")
    print("\nData collection finished successfully!")


if __name__ == "__main__":
    asyncio.run(main())
