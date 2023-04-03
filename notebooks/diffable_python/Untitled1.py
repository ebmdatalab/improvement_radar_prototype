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

from ebmdatalab import bq
import pandas as pd
import os
import json
import requests
from pandas.io.json import json_normalize
from bs4 import BeautifulSoup as bs

sql = """
SELECT table_id, date(TIMESTAMP_MILLIS(last_modified_time)) as last_modified_date 
FROM `ebmdatalab.measures.__TABLES__`
WHERE table_id LIKE 'ccg_data_%'
"""
tables_df = bq.cached_read(sql,use_cache=False,csv_path='data/tables_df.csv')

pd.set_option('display.max_rows', 200)

print(tables_df)

res = requests.get('https://github.com/ebmdatalab/openprescribing/tree/main/openprescribing/measure_definitions')    
soup = bs(res.text, 'html.parser')   
file = soup.find_all('a',class_="js-navigation-open")
json_df = pd.DataFrame()
for i in file:
    if '.json' in i.text:
        url = 'https://raw.githubusercontent.com/ebmdatalab/openprescribing/main/openprescribing/measure_definitions/' + i.text
        data = json.loads(requests.get(url).text)
        norm_df = pd.json_normalize(data, max_level=1).applymap(lambda x: x[0] if isinstance(x, list) else x)
        measure_name = i.text.replace(".json", "")
        norm_df['measure_name']= 'ccg_data_' + measure_name
        json_df = pd.concat([json_df,norm_df], axis=0, ignore_index=True)

new = json_df[['measure_name']]

print(new)

new[~new.measure_name.isin(tables_df.table_name)]



fossil_df = tables_df.merge(new, left_on='table_id', right_on='measure_name', how='outer')
fossil_df = fossil_df[fossil_df['measure_name'].isnull()]
mask = fossil_df ['table_id'].str.contains('preview', case=False, na=False)
fossil_df = fossil_df[~mask]

print(fossil_df)


