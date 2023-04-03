# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Practice level change detection for all OpenPrescribing measures

from ebmdatalab import bq
from change_detection import functions as chg
import os
from lib.outliers import *
import json
import requests
from pandas.io.json import json_normalize
from bs4 import BeautifulSoup as bs
# %load_ext autoreload
# %autoreload 2
import time

# ## Get names of "core" openprescribing measures from GitHub

#by scraping the details off of the measures definition GitHub page, we can populate graphs with titles, rather than measure name
res = requests.get('https://github.com/ebmdatalab/openprescribing/tree/main/openprescribing/measure_definitions') #open scraper for measure definitions
soup = bs(res.text, 'html.parser')   
file = soup.find_all('a',class_="js-navigation-open")
json_df = pd.DataFrame() #create dataframe 
for i in file:
    if '.json' in i.text: #find links in measure definition page for JSON measures
        url = 'https://raw.githubusercontent.com/ebmdatalab/openprescribing/main/openprescribing/measure_definitions/' + i.text #create URL for each measure definition
        data = json.loads(requests.get(url).text) #load JSON from measure definition
        norm_df = pd.json_normalize(data, max_level=1) #normalise data to fit in dataframe format and put into dataframe
        measure_name = i.text.replace(".json", "") # creates measure name from url (minus JSON)
        norm_df['measure_name'] = measure_name # adds measure name to dataframe
        json_df = pd.concat([json_df,norm_df], axis=0, ignore_index=True) #adds row to final dataframe
name_df = json_df[['measure_name','name']] #creates new df with just measure_name and descriptive name

# ## Run Change Detection module on all data at practice level

lp = chg.ChangeDetection('practice_data_%', measure=True) #ccg_data_ will run all current measures in the database
lp.run()

# ## Create one single dataframe for all change detection calculations

lp_spark  = lp.concatenate_outputs()
lp_spark.head()

# ## Create sparkline and details of change for each measure

tables_df = pd.read_csv('data/ccg_data_/measure_list.csv') #open measure list from import
tables_df = tables_df.merge(name_df, how = 'left', left_on='table_id', right_on='ccg_data_' + name_df['measure_name']) #merge with title names from JSON scraper
tables_df.head()
for index, row in tables_df.iterrows(): # loop through each row to create sparklines
    current_measure = row["table_id"] #define the current measure for loop
    current_name = row["name"] #define the current measure name for loop
    current_dir = 'ccg_data_/'+ current_measure #define the current directory for the loop
    print(current_name) #print measure name
    graph = filtered_sparkline(lp_spark, #create sparkline
                               current_dir,
                               current_measure)
    display(graph) #display graph - neccesary as in a for loop
