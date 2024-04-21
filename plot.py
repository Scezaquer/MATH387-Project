import json
import numpy as np
import matplotlib.pyplot as plt

# Load data from sgd.json
with open('sgd.json', 'r') as f:
    sgd_data = json.load(f)

# Load data from stochastic_ine_search.json
with open('sgd_quickprop_hybrid.json', 'r') as f:
    stochastic_ine_search_data = json.load(f)

# Convert data to numpy arrays
sgd_data = np.array(sgd_data)
stochastic_ine_search_data = np.array(stochastic_ine_search_data)

# Calculate average, minimum, and maximum for each trial
sgd_avg = np.mean(sgd_data, axis=0)
sgd_min = np.min(sgd_data, axis=0)
sgd_max = np.max(sgd_data, axis=0)

stochastic_avg = np.mean(stochastic_ine_search_data, axis=0)
stochastic_min = np.min(stochastic_ine_search_data, axis=0)
stochastic_max = np.max(stochastic_ine_search_data, axis=0)

# Plot average lines with shaded regions
plt.plot(sgd_avg, label='SGD')
plt.fill_between(range(len(sgd_avg)), sgd_min, sgd_max, alpha=0.3)

plt.plot(stochastic_avg, label='Quickprop-SGD hybrid')
plt.fill_between(range(len(stochastic_avg)), stochastic_min, stochastic_max, alpha=0.3)

# Add vertical line and label
plt.axvline(x=35, color='k', linestyle='--', label='Switch from QuickProp to SGD')
#plt.text(35, max(max(sgd_max), max(stochastic_max)), 'Switch from QuickProp to SGD', rotation=90, verticalalignment='top')

plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.title('Average Training Accuracy with Variability')
plt.legend()
plt.grid(True)
plt.show()
