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
import re
from pathlib import Path
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
from change_detection import functions as chg
from IPython.display import Markdown, display, Image, HTML, JSON
import matplotlib.pyplot as plt

# # %matplotlib inline


DATA_FOLDER = Path("data/ccg_data_")
GITHUB_API_URL = 'https://api.github.com'
REPO_OWNER = 'ebmdatalab'
REPO_NAME = 'openprescribing'
PATH = 'openprescribing/measure_definitions'
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

def get_measures_info(url, use_cache=True):
    """Get measure information from a given URL with optional caching."""
    measure_cache_path = DATA_FOLDER / "measure_info_cache.csv"
    
    if use_cache:
        cached_data = load_cache(measure_cache_path)
        if cached_data is not None:
            return cached_data

    file_list = fetch_from_url(url)
    if not file_list:
        return pd.DataFrame()

    filtered_files = filter_json_files(file_list)
    measure_names = extract_measure_names(filtered_files)

    name_df = pd.DataFrame.from_dict(measure_names, orient='index', columns=['name'])
    name_df = name_df.reset_index()
    name_df.columns = ["measure_name", "name"]

    cache_data(name_df, measure_cache_path)

    return name_df

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

def add_vline(ax, x_index, **kwargs):
    """Add a vertical line to the axes."""
    ax.axvline(x=x_index, **kwargs)
    return ax

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

def plot_org_data(org, data_lidocaine, deciles_lidocaine, cd_data_lidocaine, show_cd=True):
    """Plot data for an organization along with deciles, medians, and change detection."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot percentiles
    plot_percentiles(ax, deciles_lidocaine)
    
    # Plot organization data
    df_subset = data_lidocaine[data_lidocaine["code"] == org]
    df_subset = df_subset.sort_values(by="month")
    ax.plot(df_subset["month"], df_subset["rate"], linewidth=2, color="red", label="Organisation Rate")
    
    # Plot change detection data if requested
    if show_cd:
        cd_data_subset = cd_data_lidocaine[cd_data_lidocaine["name"] == org]
        initial_level, final_level, prop, drop_period = cd_data_subset.iloc[0][
            ["is.intlev.initlev", "is.intlev.finallev", "is.intlev.levdprop", "is.tfirst.big"]
        ]
        dates = df_subset["month"].sort_values()
        add_vline(ax, dates.iloc[int(drop_period)], color='green', linestyle='--', label="Change detected")
        ax.axhline(y=initial_level, color='orange', label='_nolegend_')
        ax.axhline(y=final_level, color='orange', label='_nolegend_')
        ax.fill_between(dates, initial_level, final_level, color='orange', alpha=0.5, label="Detected Change")
        print(f"Measured % change: {round(prop * 100, 1)}%")

    ax.set_ylabel('Rate', size=18)
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
    'show_cd': True,
    'show_filter_results': True,
    'goes_down': True,
    'goes_down_threshold': 0,
    'drop_starts_high': True,
    'drop_starts_high_quantile': 0.9,
    'drop_ends_high': True,
    'drop_ends_high_quantile': 0.9,
    'apply_high_filter': True,
    'apply_mean_difference_filter': True,
    'mean_difference_threshold': 0.4,
    'apply_zero_filter': True,
    'apply_mean_events_filter': True,
    'mean_events_threshold': 0,
    'top_x': 5
}


class DataFrameProcessor:
    def __init__(self, df_main, df_secondary, config=None, verbose=False):
        self.df_main = df_main.sort_values(by="month", ascending=True)
        self.df_secondary = df_secondary
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)    
        self.verbose = verbose
    
    def _apply_threshold_mask(self, column, threshold_key):
        threshold = self.config.get(threshold_key)
        return self.df_secondary[column] > threshold
    
    def _apply_quantile_threshold_mask(self, column, quantile_key):
        threshold = self.df_secondary[column].quantile(self.config.get(quantile_key))
        return self.df_secondary[column] < threshold

    
    def _print_before_after(self, mask, description):
        if self.verbose:
            print(f"{description} - Before: {self.df_secondary.shape[0]}")
        
        self.df_secondary = self.df_secondary[mask]
        
        if self.verbose:
            print(f"{description} - After: {self.df_secondary.shape[0]}")
            print('---------')
        return self
     
    def _goes_down_threshold(self):
        return self._print_before_after(self._apply_threshold_mask('is.intlev.levdprop', 'goes_down_threshold'), "CD: Decected proportional drop < threshold")
    
    def _drop_starts_high(self):
        return self._print_before_after(self._apply_quantile_threshold_mask('is.intlev.initlev', 'drop_starts_high_quantile'), "CD: Detected change initial level > threshold (quantile)")
    
    def _drop_ends_high(self):
        return self._print_before_after(self._apply_quantile_threshold_mask('is.intlev.finallev', 'drop_ends_high_quantile'), "CD: Detected change final level > threshold (quantile)")
    
    def _get_filtered_df(self):
        if self.config.get('goes_down', False):
            self._goes_down_threshold()
        if self.config.get('drop_starts_high', False):
            self._drop_starts_high()
        if self.config.get('drop_ends_high', False):
            self._drop_ends_high()
        return self.df_secondary
        
    
    def _compute_mean_for_series_slice(self, series, start=None, end=None):
        """Compute the mean for a slice of a series."""
        return series.iloc[start:end].mean()

    def _has_high_percentiles(self, series):
        first_six_mean = self._compute_mean_for_series_slice(series, end=6)
        last_six_mean = self._compute_mean_for_series_slice(series, start=-6)
        return ((first_six_mean > 0.8) & (last_six_mean > 0.8))

    def _mean_difference(self, series):
        first_six_mean = self._compute_mean_for_series_slice(series, end=6)
        last_six_mean = self._compute_mean_for_series_slice(series, start=-6)
        return last_six_mean - first_six_mean

    def _goes_to_zero(self, series):
        last_six_mean = self._compute_mean_for_series_slice(series, start=-6)
        return last_six_mean == 0

    def _mean_events(self, series):
        return self._compute_mean_for_series_slice(series)


    def _get_orgs_not_remaining_high(self):
        remains_high = self.df_main.groupby('code')['percentile'].transform(self._has_high_percentiles)
        return set(self.df_main[remains_high]['code'])

    def _get_mean_differences(self):
        return self.df_main.groupby('code')['percentile'].transform(self._mean_difference)
    
    def _get_mean_events(self):
        return self.df_main.groupby('code')['numerator'].transform(self._mean_events)
    
    def _get_orgs_not_going_to_zero(self):
        goes_to_zero = self.df_main.groupby('code')['rate'].transform(self._goes_to_zero)
        return set(self.df_main[goes_to_zero]['code'])


    def _filter_orgs(self, orgs):
        """Filter out orgs that don't remain high."""
        orgs_not_remaining_high = self._get_orgs_not_remaining_high()
        return [org for org in orgs if org not in orgs_not_remaining_high]
    
    def _filter_orgs_by_mean_difference(self, orgs, threshold=0):
        """Filter orgs based on mean differences greater than the given threshold."""
        mean_diffs = self._get_mean_differences()
  
        valid_orgs = self.df_main.loc[mean_diffs < -abs(threshold), 'code'].unique()
        return [org for org in orgs if org in valid_orgs]
    
    def _filter_orgs_going_to_zero(self, orgs):
        orgs_not_going_to_zero = self._get_orgs_not_going_to_zero()
        return [org for org in orgs if org not in orgs_not_going_to_zero]
    
    def _filter_orgs_by_low_events(self, orgs, threshold=100):
        """Filter orgs based on mean differences greater than the given threshold."""
        mean_events = self._get_mean_events()
  
        valid_orgs = self.df_main.loc[mean_events > threshold, 'code'].unique()
        return [org for org in orgs if org in valid_orgs]
    
    def _print_change_counts(self, initial_count, filtered_orgs, description):
        """Print the number of items before and after a filter is applied."""
        if self.verbose:
            print(f"{description} - Before: {initial_count}")
            print(f"{description} - After: {len(filtered_orgs)}")
            print('---------')

    def apply_combined_filters(self):
        """Apply configured filters to the list of orgs."""
        initial_count = len(self.df_secondary)
        self._get_filtered_df()
        
        filtered_orgs = self.df_secondary['name'].unique()

        if self.config.get('apply_high_filter', False):
            initial_count = len(filtered_orgs)
            filtered_orgs = self._filter_orgs(filtered_orgs)
            self._print_change_counts(initial_count, filtered_orgs, "Remains in high percentile")

        if self.config.get('apply_mean_difference_filter', False):
            threshold = self.config.get('mean_difference_threshold', 0)
            initial_count = len(filtered_orgs)
            filtered_orgs = self._filter_orgs_by_mean_difference(filtered_orgs, threshold)
            self._print_change_counts(initial_count, filtered_orgs, "Doesn't drop > percentile threshold")
        
        if self.config.get('apply_mean_events_filter', False):
            threshold = self.config.get('mean_events_threshold', 0)
            initial_count = len(filtered_orgs)
            filtered_orgs = self._filter_orgs_by_low_events(filtered_orgs, threshold)
            self._print_change_counts(initial_count, filtered_orgs, "Low average number of events")
            
        if self.config.get("apply_zero_filter", False):
            initial_count = len(filtered_orgs)
            filtered_orgs = self._filter_orgs_going_to_zero(filtered_orgs)
            self._print_change_counts(initial_count, filtered_orgs, "Drops to zero")
        
        # order by
        self.df_secondary = self.df_secondary.loc[self.df_secondary["name"].isin(filtered_orgs)]
      
        filtered_orgs = self.df_secondary['name'].unique().tolist()[:self.config.get("top_x")]
        
        return filtered_orgs


# -

config = {
    'show_cd': False,
    'show_filter_results': False,
    'goes_down': True,
    'goes_down_threshold': 0.2,
    'drop_starts_high': True,
    'drop_starts_high_quantile': 0.8,
    'drop_ends_high': True,
    'drop_ends_high_quantile': 0.4,
    'apply_high_filter': True,
    'apply_mean_difference_filter': True,
    'mean_difference_threshold': 0.2,
    'apply_zero_filter': True,
    'apply_mean_events_filter': True,
    'mean_events_threshold': 20,
    'top_x': 5
}

# +
ccg_names = pd.read_csv("data/ccg_names.csv")
measure_list = pd.read_csv(DATA_FOLDER / 'measure_list.csv', usecols=['table_id'])

measure_info = get_measures_info(CONTENT_URL)

measure_list = measure_list.merge(measure_info, how = 'left', left_on='table_id', right_on='ccg_data_' + measure_info['measure_name']) #merge with title names from JSON scraper
measures = [m.replace("ccg_data_", "") for m in measure_list["table_id"]]
# -

# # OpenPrescribing Improvement Radar
#
# ## What this tool does
# This tool identifies sub-ICB locations (SICBLs) which have shown substantial improvement across each of our OpenPrescribing measures. The five SICBLs with the largest improvement are reported. We hope this will stimulate discussion with areas that have made effective changes so that successful strategies can be shared.
#
# ## How it works
# We used trend indicator saturation to detect the timing and size of changes of SICBLs against all SICBLs in England. This prototype uses the current criteria to identify improvement:
# * SICBLs needed to be in the highest 20% at the start of the change
# * SICBLs needed to improve to the lowest 40% of SICBLs at the end of the change
# * SICBLs needed to improve by at least 20 percentiles
# * There needed to be at least 20 prescription items written
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

for m in measures:
    measure_link = f"https://openprescribing.net/measure/{m}"
    measure_description = measure_list.loc[measure_list["measure_name"] == m, "name"]
    if len(measure_description) > 0:
        measure_description=measure_description.iloc[0]
    
        display(Markdown(f'<a href=#{m}>- {measure_description}</a>'))
        
        
for m in measures:
    
    measure_link = f"https://openprescribing.net/measure/{m}"
    measure_description = measure_list.loc[measure_list["measure_name"] == m, "name"]
    if len(measure_description) > 0:
        measure_description=measure_description.iloc[0]
        
        display(Markdown(f'<h2 id={m}><a href={measure_link}>{measure_description}</a></h2>'))
        data = pd.read_csv(DATA_FOLDER / f'ccg_data_{m}/bq_cache.csv', parse_dates=["month"])
        data["rate"] = data["numerator"]/data["denominator"]

        cd_data = pd.read_csv(DATA_FOLDER / f'ccg_data_{m}/r_output.csv')


        data["percentile"] = data.groupby(["month"])["rate"].rank(pct=True)
    
        deciles = compute_deciles(data, "month", "rate")

        processor = DataFrameProcessor(data, cd_data, config, verbose=config.get('show_filter_results', False))
        filtered_ccgs = processor.apply_combined_filters()


        num_orgs_identified = len(filtered_ccgs)
        # Main loop to plot CCG data
        if num_orgs_identified > 0:
#             display(Markdown(f"Number of organisations with improvement identified: {num_orgs_identified}"))
            for ccg in filtered_ccgs:
                sicbl_link = f"https://openprescribing.net/measure/{m}/sicbl/{ccg}"
                ccg_name = ccg_names.loc[ccg_names["code"]==ccg, "name"].values[0]
                display(Markdown(f'<h4><a href={sicbl_link}>{ccg_name}</a></h4>'))
                fig = plot_org_data(ccg, data, deciles, cd_data, show_cd=config.get('show_cd', False))
                
             
        else:
            display(Markdown("No organisations met the technical criteria for detecting substantial change on this measure."))
# +


