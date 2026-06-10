# H1 UK Parliament Petitions Overview Dashboard
## H2 Project description


## H2 Key files
* daily web scraping.py: This script scrapes the UK Parliament petitions website for data on all open petitions (see below for more detail). It runs every day at 6:00am UTC [Note: this is 7:00am in the UK during BST and 6:00am outside BST]. Data pulls from each day are stored in an Amazon S3 bucket.
* dashboard.py: This script uses the above-mentioned data as well as other sources to produce a dashboard, providing a constituency-level overview of open petitions.


## H2 Daily web scraping outputs
Below I set out all the variables that are scraped as part of the daily web scraping script. The data is organised into two separate datasets: petitions_list_{date_of_datapull} and petitions_counts_{date_of_datapull}. For example, data pulled on 1 June 2026 will be saved into one file called petitions_list_20260601.csv and another called petitions_counts_20260601.csv


## H3 Variables in petitions_list_{date_of_datapull}.csv
| Variable name | Variable description | Data type | Other notes |
|----------|----------|----------|----------|
| `petition_title` | Name of petition | Free text | |
| `petition_url` | URL for petition main page | HTML | |
| `petition_status` | Status of petition | Binary variable: 'open' or 'closed' | |
| `total_signature_count` | Overall total number of signatures (all constituencies combined) | Integer | |
| `petition_id` | Unique identifier for each petition, created by UK Parliament website | 6 digit number | |
| `opened_at` | Date petition opened | Date | This should be populated for all petitions |
| `debate_threshold_reached_at` | Date the petition reached 100,000 signatures in total | Date | NA if petition has fewer than 100,000 signatures. |
| `scheduled_debate_date` | Date petition debate is scheduled to take place | Date | NA if petition has fewer than 100,000 signatures or has not yet had a date set for debate |
| `deadline` | Date petition is scheduled to close | Date | This should be 6 months after opened_at date. |


## H3 Variables in petitions_counts_{date_of_datapull}.csv
| Variable name | Variable description | Data type | Other notes |
|----------|----------|----------|----------|
| `petition_id`| Unique identifier for each petition, created by UK Parliament website | 6 digit number | |
| `PCON24CD` | Unique identifier for each constituency | E or W or S or N followed by 8 digits | |
| `constituency_name` | Constituency name | Free text | |
| `signature_count` | Number of signatures in each constituency | Integer | Lowest possible value is 1. If true count is 0, there will be no row for it. |


## H2 Other data sources
As noted above, the dashboard uses data from other sources as well.
| Data description | File type | Data source | Version | Use |
|----------|----------|----------|----------|
| Westminster Parliamentary Constituencies (July 2024) Names and Codes in the UK (V2)
| geojson
| [ONS, Westminster Parliamentary Constituencies July 2024] (https://www.data.gov.uk/dataset/ceccb29c-3a8c-4d4e-a1eb-f3088dfc8cc6/westminster-parliamentary-constituencies-july-2024-boundaries-uk-bfe1)
| Downloaded on 9 June 2026
| To create choropleth |

| Constituency population estimates
| csv
| [NOMIS, Population estimates - local authority based by single year of age] (https://www.nomisweb.co.uk/datasets/pestsyoala)
| Downloaded on 7 June 2026
| To calculate signatures per 1000 population |    

