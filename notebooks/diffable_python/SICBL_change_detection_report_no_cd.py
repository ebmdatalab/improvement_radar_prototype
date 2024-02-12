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
#   vscode:
#     interpreter:
#       hash: 692be894feba73f0646533f6ae9a20606d08b65b9e0f7c2d28298b63f25cbb72
# ---

# +
import requests
from pathlib import Path
import pandas as pd
import numpy as np
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
from ebmdatalab import bq
import os
import json
from pandas import json_normalize
import shutil
import matplotlib.ticker as mtick

DATA_FOLDER = Path("data/ccg_data_")
GITHUB_API_URL = 'https://api.github.com'
REPO_OWNER = 'ebmdatalab'
REPO_NAME = 'openprescribing'
PATH = 'openprescribing/measures/definitions'
CONTENT_URL = f'{GITHUB_API_URL}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{PATH}'


# +
def fetch_from_url(url):
    """Fetch the content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None

def filter_json_files(file_list):
    """Filter files to only include those with a .json extension."""
    return [f for f in file_list if f['name'].endswith('.json')]

def extract_measure_names(file_list):
    """Extract measure names from a list of files."""
    measure_names = {}
    for file in file_list:
        name = file['name'].replace('.json', '')
        content_url = file['download_url']
        content_data = fetch_from_url(content_url)
        if content_data:
            measure_name = content_data.get('name', '')
            measure_names[name] = measure_name
    return measure_names

def cache_data(df, cache_path):
    """Cache the data to a CSV file."""
    df.to_csv(cache_path, index=False)

def load_cache(cache_path):
    """Load cached data from a CSV file if it exists."""
    if cache_path.exists():
        return pd.read_csv(cache_path)
    return None

def compute_deciles(measure_table, groupby_col, values_col, has_outer_percentiles=False):
    """
    Computes deciles.

    Args:
        measure_table (pd.DataFrame): A measure table.
        groupby_col (str): The name of the column to group by.
        values_col (str): The name of the column for which deciles are computed.
        has_outer_percentiles (bool, optional): Whether to compute the nine largest and nine smallest
            percentiles as well as the deciles. Defaults to False.

    Returns:
        pd.DataFrame: A data frame with `groupby_col`, `values_col`, and `percentile` columns.
    """
    
    quantiles = np.linspace(0.1, 0.9, 9)
    
    if has_outer_percentiles:
        lower_percentiles = np.linspace(0.01, 0.09, 9)
        upper_percentiles = np.linspace(0.91, 0.99, 9)
        quantiles = np.concatenate([lower_percentiles, quantiles, upper_percentiles])
    
    percentiles = (
        measure_table.groupby(groupby_col)[values_col]
        .quantile(quantiles)
        .reset_index()
    )
    
    percentiles["percentile"] = (percentiles["level_1"] * 100).astype(int)
    percentiles = percentiles[[groupby_col, "percentile", values_col]]
    
    return percentiles

def plot_percentiles(ax, deciles_lidocaine):
    """Plot percentiles on the given axes."""
    linestyles = {
        "decile": {"line": "b--", "linewidth": 1, "label": "Decile"},
        "median": {"line": "b-", "linewidth": 1.5, "label": "Median"},
        "percentile": {"line": "b:", "linewidth": 0.8, "label": "1st-9th, 91st-99th percentile"},
    }
    label_seen = []

    for percentile in range(1, 100):
        data = deciles_lidocaine[deciles_lidocaine["percentile"] == percentile]
        style = linestyles["median"] if percentile == 50 else linestyles["decile"]
        label = style["label"] if percentile == 50 or "decile" not in label_seen else "_nolegend_"
        if "decile" not in label_seen and percentile != 50:
            label_seen.append("decile")
        ax.plot(data["month"], data["rate"], style["line"], linewidth=style["linewidth"], label=label)

def plot_org_data(org, data, deciles, percentage, y_string):
    """Plot data for an organization along with deciles, medians, and change detection."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot percentiles
    plot_percentiles(ax, deciles)
    
    # Plot organization data
    df_subset = data[data["code"] == org]
    df_subset = df_subset.sort_values(by="month")
    ax.plot(df_subset["month"], df_subset["rate"], linewidth=2, color="red", label="Organisation Rate")
    
    ax.set_ylabel(y_string, size=14)
    if percentage: 
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0)) #  if percentage then use percentage formatter
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=14)
    ax.margins(x=0)
    plt.tight_layout()
    plt.show()
    plt.close(fig)



# +
DEFAULT_CONFIG = {
    'show_filter_results': False,
    'apply_start_end_percentile_filter': True,
    'start_percentile_threshold': 0.8,
    'end_percentile_threshold': 0.4,
    'apply_rate_decrease_filter': True,
    'rate_decrease_percent': 10,
    'apply_zero_filter': True,
    'apply_mean_events_filter': True,
    'mean_events_threshold': 50,
    'top_x_orgs': None,
}


class DataFrameProcessor:
    def __init__(self, df_main, config=None, verbose=False):
        self.df_main = df_main.sort_values(by="month")
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.verbose = verbose

    def _print_change_counts(self, description, before, after):
        """Print the number of items before and after a filter is applied."""
        if self.verbose:
            print(f"{description} - Before: {before}, After: {after}\n---------")

    def _apply_filters(self):
        """Apply all filters based on the configuration."""
        filters = [
            ("apply_rate_decrease_filter", self._filter_by_rate_decrease),
            ("apply_mean_events_filter", self._filter_by_mean_events),
            ("apply_zero_filter", self._filter_going_to_zero),
            ("apply_start_end_percentile_filter", self._filter_by_start_end_percentile),
        ]

        filtered_df = self.df_main
        for conf_key, filter_func in filters:
            if self.config.get(conf_key):
                initial_count = filtered_df['code'].nunique()
                filtered_df = filter_func(filtered_df)
                if filtered_df.empty:
                    break
                self._print_change_counts(conf_key, initial_count, filtered_df['code'].nunique())
        
        top_x = self.config.get('top_x_orgs')
        if top_x:
            filtered_df = self._select_top_x_by_percentile_drop(filtered_df, top_x)

        self.df_main = filtered_df

    def _filter_going_to_zero(self, df):
        """Filter out orgs whose rates go to zero in the last six months."""
        goes_to_zero = df.groupby('code')['rate'].transform(lambda x: x.iloc[-6:].mean() == 0)
        return df[~goes_to_zero]

    def _filter_by_mean_events(self, df):
        """Filter orgs by a minimum threshold of mean events."""
        threshold = self.config.get('mean_events_threshold', 20)
        mean_events = df.groupby('code')['numerator'].transform('mean')
        return df[mean_events > threshold]

    def _filter_by_rate_decrease(self, df):
        """Filter orgs where the rate decreases by at least a certain percentage, handling division by zero."""
        decrease_percent = self.config.get("rate_decrease_percent", 10) / 100

        def calculate_rate_change(x):
            start_mean = x.iloc[:6].mean()
            end_mean = x.iloc[-6:].mean()
            
            if start_mean == 0:
                return None
            return (start_mean - end_mean) / start_mean

        rate_change = df.groupby('code')['rate'].apply(calculate_rate_change)
        valid_codes = rate_change.dropna()[rate_change >= decrease_percent].index
        return df[df['code'].isin(valid_codes)]

    def _filter_by_start_end_percentile(self, df):
        """Filter orgs that start in a high percentile and end in a low percentile."""
        start_threshold = self.config.get('start_percentile_threshold', 0.8)
        end_threshold = self.config.get('end_percentile_threshold', 0.4)
        
        start_percentile = df.groupby('code').apply(lambda x: x['percentile'].iloc[:6].mean())
        end_percentile = df.groupby('code').apply(lambda x: x['percentile'].iloc[-6:].mean())
        valid_orgs = start_percentile[start_percentile > start_threshold].index.intersection(end_percentile[end_percentile < end_threshold].index)
        return df[df['code'].isin(valid_orgs)]
    
    def _select_top_x_by_percentile_drop(self, df, top_x):
        """Selects the top x organizations based on the largest percentile drop."""
        
        start_percentile = df.groupby('code')['percentile'].transform(lambda x: x.iloc[:6].mean())
        end_percentile = df.groupby('code')['percentile'].transform(lambda x: x.iloc[-6:].mean())

        df['percentile_drop'] = start_percentile - end_percentile
        
        top_x_codes = (df[['code', 'percentile_drop']]
                       .drop_duplicates('code')
                       .sort_values(by='percentile_drop', ascending=False)
                       .head(top_x)['code'])

        return df[df['code'].isin(top_x_codes)]

    def apply_combined_filters(self):
        self._apply_filters()
        return self.df_main['code'].unique().tolist()


# -

config = {
    'show_filter_results': False,
    'apply_start_end_percentile_filter': True,
    'start_percentile_threshold': 0.8,
    'end_percentile_threshold': 0.5,
    'apply_rate_decrease_filter': True,
    'rate_decrease_percent': 5,
    'apply_zero_filter': True,
    'apply_mean_events_filter': True,
    'mean_events_threshold': 50,
    'top_x_orgs': 5,
}

# +
#get latest measures list and refresh data
res = requests.get('https://api.github.com/repos/ebmdatalab/openprescribing/contents/openprescribing/measures/definitions') #uses GitHub API to get list of all files listed in measure definitions - defaults to main branch
data = res.text #creates text from API result
df = pd.read_json(data) #turns JSON from API result into dataframe
json_df = pd.DataFrame() #creates blank dataframe
for row in df[df['name'].str.contains('.json')].itertuples(index=True): #iterates through rows, and continues if file is .json
        url = (getattr(row, "download_url")) #gets URL from API request  
        data = json.loads(requests.get(url).text) #gets JSON from measure definition URL
        norm_df = pd.json_normalize(data, max_level=1) #normalises measure definition JSON into dataframe
        json_df = pd.concat([json_df,norm_df], axis=0, ignore_index=True) # concatentates into single dataframe
        json_df['table_id'] = 'ccg_data_' + df['name'].str.split('.').str[0].copy()
tags_df = json_df.explode('tags') # explode json so each tag is on seperate line
core_df = tags_df[['table_id', 'name', 'tags', 'radar_exclude', 'is_percentage','numerator_short', 'denominator_short']].copy() # create smaller df based on tags_df
core_df['measure_name'] = core_df['table_id'].apply(lambda x: x.replace('ccg_data_', '')) # remove `ccg_data_` from measure_name for use in URL
core_df['y_string'] = core_df['numerator_short'] + ' per ' +  core_df['denominator_short']
measure_list = core_df[((core_df['tags'].str.contains('core')) | (core_df['tags'].str.contains('lowpriority'))) & (core_df['radar_exclude'] != 'True')] #filter to only core measures where `radar_exclude` is not true
measure_list.to_csv(DATA_FOLDER / 'measure_list.csv', index=False)  #save measure_list

#create for next loop to go through each table name in the previous query
for name in measure_list['table_id']:
    
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
    bq.cached_read(sql, os.path.join(DATA_FOLDER, "{}", "bq_cache.csv").format(name), use_cache=False)
    
#delete any unused measure folders
for folder in os.listdir(DATA_FOLDER):
    if os.path.isdir(os.path.join(DATA_FOLDER, folder)):  # Check if it's a directory
        if folder not in measure_list['table_id'].values:  # Check if folder is not in DataFrame
            # Delete the folder and its contents
            folder_path = os.path.join(DATA_FOLDER, folder)
            shutil.rmtree(folder_path)
# -

#get SICBL names
sql = """
    SELECT
      code, 
      name 
    FROM
      ebmdatalab.hscic.ccgs
    WHERE
      name IS NOT NULL
    GROUP BY
      code, 
      name     
    """
ccg_names = bq.cached_read(sql, os.path.join(DATA_FOLDER, "ccg_names.csv"), use_cache=False)

# # OpenPrescribing Improvement Radar
#
# ## What this tool does
# This tool identifies sub-ICB locations (SICBLs) which have shown substantial improvement across each of our OpenPrescribing measures. The five SICBLs with the largest improvement are reported. We hope this will stimulate discussion with areas that have made effective changes so that successful strategies can be shared.
#
# ## How it works
# We used trend indicator saturation to detect the timing and size of changes of SICBLs against all SICBLs in England. This prototype uses the current criteria to identify improvement:
# * SICBLs needed to, on average, be in the highest 20% during the first 6 months of the time period shown.
# * SICBLs needed to improve to be, on average, in the lowest 50% of SICBLs during the last 6 months of the period shown.
# * The rate of the measure has to decrease by at least 5% across the time period shown (calculated as the difference between the average rate for the first 6 months and the last 6 months). 
# * There needed to be, on average, at least 50 prescription items written.
#
# You can find more information on the trend indicator saturation methodology we use [here](https://www.bmj.com/content/367/bmj.l5205), including a podcast on our work with Professor Ben Goldacre.
#
# ## Interpretation notes
# These pilot results are provided for the interest of advanced users, although we don't know how relevant they are in practice. There is substantial variation in prescribing behaviours, across various different areas of medicine. Some variation can be explained by demographic changes, or local policies or guidelines, but much of the remaining variation is less easy to explain.
#
# We are keen to hear your feedback on this tool and how you use it. You can do this by emailing us at bennett@phc.ox.ac.uk. Please do not include patient identifiable information in your feedback.
#

# +
display(Markdown('## Table of Contents'))

measure_list = measure_list.sort_values(by='name') # sort by name
for m in measure_list['measure_name']:
    measure_link = f"https://openprescribing.net/measure/{m}"
    measure_description = measure_list.loc[measure_list["measure_name"] == m, "name"]
    if len(measure_description) > 0:
        measure_description=measure_description.iloc[0]
    
        display(Markdown(f'<a href=#{m}>- {measure_description}</a>'))
        
        
for m in measure_list['measure_name']:
    
    measure_link = f"https://openprescribing.net/measure/{m}"
    measure_description = measure_list.loc[measure_list["measure_name"] == m, "name"]
    percentage = measure_list.loc[measure_list["measure_name"] == m, "is_percentage"].item() # create percentage flag for formatting
    y_string = measure_list.loc[measure_list["measure_name"] == m, "y_string"].item() # create percentage flag for formatting
    if len(measure_description) > 0:
        measure_description=measure_description.iloc[0]
        
        display(Markdown(f'<h2 id={m}><a href={measure_link}>{measure_description}</a></h2>'))
        data = pd.read_csv(DATA_FOLDER / f'ccg_data_{m}/bq_cache.csv', parse_dates=["month"])
        data["rate"] = data["numerator"]/data["denominator"]

        data["percentile"] = data.groupby(["month"])["rate"].rank(pct=True)
    
        deciles = compute_deciles(data, "month", "rate")

        processor = DataFrameProcessor(data, config, verbose=config.get('show_filter_results', False))
        filtered_ccgs = processor.apply_combined_filters()


        num_orgs_identified = len(filtered_ccgs)
        # Main loop to plot CCG data
        if num_orgs_identified > 0:
#             display(Markdown(f"Number of organisations with improvement identified: {num_orgs_identified}"))
            for ccg in filtered_ccgs:
                sicbl_link = f"https://openprescribing.net/measure/{m}/sicbl/{ccg}"
                ccg_name = ccg_names.loc[ccg_names["code"]==ccg, "name"].values[0]
                display(Markdown(f'<h4><a href={sicbl_link}>{ccg_name}</a></h4>'))
                fig = plot_org_data(ccg, data, deciles, percentage, y_string)
                
             
        else:
            display(Markdown("No organisations met the technical criteria for detecting substantial change on this measure."))
# -


