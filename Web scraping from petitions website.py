#!/usr/bin/env python
# coding: utf-8

# ## Web scraping from UK petitions website

# In[4]:


# Importing libraries
from bs4 import BeautifulSoup
import boto3
from IPython.display import HTML
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
load_dotenv()

pd.set_option('display.max_rows', None)

today = date.today()

headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"}


# ### Downloading list of all petitions closed in the last 12 months

# In[6]:


## Setting up ##
url = 'https://petition.parliament.uk/petitions.csv?state=closed'

response = requests.get(url, headers=headers)
response.raise_for_status()

all_closed_petitions = pd.read_csv(StringIO(response.text))


## Cleaning ##
# Renaming columns
all_closed_petitions = all_closed_petitions.rename(columns = {
    'Petition': 'petition_title',
    'URL': 'petition_url',
    'State': 'status',
    'Signatures Count': 'total_signature_count'
})

# Extracting petition ID from URL
all_closed_petitions['petition_id'] = all_closed_petitions['petition_url'].str.split('/').str[-1]


# In[7]:


# Creating function to extract relevant dates for closed petitions
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
            return None, None
    
    async with aiohttp.ClientSession() as session:
        tasks = [open_and_closed_dates(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Using above function
results = await fetch_dates_for_closed_petitions(all_closed_petitions['petition_url'])

# Adding all dates into column
all_closed_petitions['opened_at'] = [r[0] for r in results]
all_closed_petitions['debate_threshold_reached_at'] = [r[1] for r in results]
all_closed_petitions['scheduled_debate_date'] = [r[2] for r in results]
all_closed_petitions['closed_at'] = [r[3] for r in results]


# In[8]:


# Restricting to petitions that closed in the last 12 months
cut_off_date = (datetime.now() - relativedelta(months=12)).date()
recent_closed_petitions = all_closed_petitions[all_closed_petitions['closed_at'] >= cut_off_date]


# In[9]:


recent_closed_petitions.head()


# In[10]:


# Creating function to extract data on number of signatures in each constituency
async def fetch_petition_data(session, url, petition_id, headers):
    try:
        async with session.get(f"{url}.json", headers=headers) as response:
            # Ensure that the response is successful
            if response.status == 200:
                data = await response.json()
                # Process and return the petition data
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

# Creating function to run above function
async def run(df):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_petition_data(session, petition['petition_url'], petition['petition_id'], headers)
            for _, petition in df.iterrows()
        ]
        all_petitions_counts = await asyncio.gather(*tasks)

    # Flatten the list of results into a single list
    return pd.DataFrame([item for sublist in all_petitions_counts for item in sublist])

# Running functions
if __name__ == "__main__":
    start_time = time.time()
    
    try:
        closed_petition_counts_df = await run(recent_closed_petitions)
    except RuntimeError as e:
        print(f"Error: {e}")

    # Print the time taken to execute
    print(f"Total time: {time.time() - start_time:.2f} seconds")


# In[11]:


closed_petition_counts_df.head()


# ### Downloading list of all open petitions

# In[13]:


## Setting up ##
url = 'https://petition.parliament.uk/petitions.csv?state=open'

response = requests.get(url, headers=headers)
response.raise_for_status()

open_petitions = pd.read_csv(StringIO(response.text))


## Cleaning ##
# Renaming columns
open_petitions = open_petitions.rename(columns = {
    'Petition': 'petition_title',
    'URL': 'petition_url',
    'State': 'status',
    'Signatures Count': 'total_signature_count'
})

open_petitions['petition_id'] = open_petitions['petition_url'].str.split('/').str[-1]


# In[14]:


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
                
                opened = None
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
            return None, None
    
    async with aiohttp.ClientSession() as session:
        tasks = [open_and_deadline_dates(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Usage
results = await fetch_dates_for_open_petitions(open_petitions['petition_url'])
open_petitions['opened_at'] = [r[0] for r in results]
open_petitions['deadline'] = [r[1] for r in results]
open_petitions['debate_threshold_reached_at'] = [r[2] for r in results]
open_petitions['scheduled_debate_date'] = [r[3] for r in results]


# In[15]:


open_petitions.head()


# In[16]:


# Check if running in Jupyter or a normal Python script environment
if __name__ == "__main__":
    start_time = time.time()
    try:
        open_petition_counts_df = await run(open_petitions)
    except RuntimeError as e:
        print(f"Error: {e}")

    # Print the time taken to execute
    print(f"Total time: {time.time() - start_time:.2f} seconds")


# ## Merging all datasets together

# In[18]:


# Merging closed and open datasets together
all_petitions_list = pd.concat([recent_closed_petitions, open_petitions], ignore_index=True)
all_petitions_counts = pd.concat([closed_petition_counts_df, open_petition_counts_df], ignore_index = True)


# In[19]:


# Exporting to Amazon workspace
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_DEFAULT_REGION')
bucket = 'uk-parliament-petitions-bucket'

s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# Creating function to export files
def upload_to_s3(df, file_name):
    csv_buffer = StringIO()  # Create an in-memory buffer
    df.to_csv(csv_buffer, index=False)  # Write the DataFrame to the buffer
    csv_buffer.seek(0)  # Go to the start of the buffer
    
    # Upload the CSV to S3
    s3_client.put_object(
        Bucket=bucket,
        Key=file_name,
        Body=csv_buffer.getvalue()
    )

# Exporting
upload_to_s3(all_petitions_list, 'dynamic data/all_petitions_list.csv')
upload_to_s3(all_petitions_counts, 'dynamic data/all_petitions_counts.csv')

