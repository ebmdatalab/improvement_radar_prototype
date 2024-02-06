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

import pandas as pd
import json
import requests
from pandas import json_normalize
#import ebmdatalab
from ebmdatalab import bq
import os

res = requests.get('https://api.github.com/repos/ebmdatalab/openprescribing/contents/openprescribing/measures/definitions') #uses GitHub API to get list of all files listed in measure definitions - defaults to main branch
data = res.text #creates text from API result
df = pd.read_json(data) #turns JSON from API result into dataframe

json_df = pd.DataFrame() #creates blank dataframe
for row in df[df['name'].str.contains('.json')].itertuples(index=True): #iterates through rows, and continues if file is .json
        url = (getattr(row, "download_url")) #gets URL from API request  
        data = json.loads(requests.get(url).text) #gets JSON from measure definition URL
        norm_df = pd.json_normalize(data, max_level=1) #normalises measure definition JSON into dataframe
        json_df = pd.concat([json_df,norm_df], axis=0, ignore_index=True) # concatentates into single dataframe
        json_df['file_name'] = 'ccg_data_' + df['name'].str.split('.').str[0].copy()

tags_df = json_df.explode('tags')
core_df = tags_df[['file_name', 'tags', 'radar_exclude']].copy()
filtered_core_df = core_df[((core_df['tags'].str.contains('core')) | (core_df['tags'].str.contains('lowpriority'))) & (core_df['radar_exclude'] != 'True')]

display(filtered_core_df)

#create for next loop to go through each table name in the previous query
for name in filtered_core_df['file_name']:
    
    sql = """
    SELECT
      month, 
      pct_id as code, 
      numerator, 
      denominator, 
    FROM
      `ebmdatalab.measures.{}` AS a
    """
    
    sql = sql.format(name) #using python string to add table_name to SQL
    #concatenate each table name into single file during for next loop
    bq.cached_read(sql, os.path.join("..", "data", "test", "{}", "bq_cache.csv").format(name), use_cache=False)


