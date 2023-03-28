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
from change_detection import functions as chg
import os
# %load_ext autoreload
# %autoreload 2

lp = chg.ChangeDetection('practice_data_carbon_salbutamol',
                         measure=True)
lp.run()

lp.concatenate_outputs()


