# uk-petitions

Scripts needed for current version of dashboard:

1. Web scraping for dashboard.py


2. Dashboard.py
This file pulls data from the S3 storage to create the dashboards and dropdowns that the user sees on the website.



Scripts created for future iterations:

1. Initial web scraping.py


2. Daily web scraping.py
This scripts run every morning at 6am GDT???

Outputs
    1. petitions_list_[today date]
* petition_title: petition title
* petition_url: petition URL
* petition_status: closed or open
* total signature count: total number of signatures (all constituencies combined)
* petition_id: unique petition id
* opened_at: should be populated for all petitions
* debate_threshold_reached_at: date petition reached XX date
* scheduled_debate_date: date petition debate is scheduled
* closed_at: populated for all closed petitions

    2. petitions_counts_[today date]
* petition_id:
* PCON24CD: unique identifier for each constituency
* constituency_name:
* signature_count: 


Static data sources:
Constituency level population estimates:
    Population estimates - small area (2021 based) by single year of age - England and Wales
    https://www.nomisweb.co.uk/query/construct/summary.asp?menuopt=200&subcomp=

    

