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

# +
import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.arange(10)
y = np.random.rand(10) * 1000
# Function to add line break to y-axis label if the label exceeds a certain length
def format_label(label):
    if len(label) > 7:  # Adjust the threshold length as needed
        split_index = len(label) // 2
        label = label[:split_index] + '\n' + label[split_index:]
    return label

# Plotting
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_ylabel(format_label('This is a long label'))  # Example long label
plt.show()
# -


