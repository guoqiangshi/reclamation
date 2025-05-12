# This script calculates the annual change rate. The results are saved to text files and can be used to analyze trends over time.

"""
Note - In this study, the calculated results is visualized by plotting the change rates every 5 years to observe trends over time.
"""

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import pandas as pd

# Define a function to calculate the slope of subsidence using linear regression
def calculate_slope(df, start_year, end_year):
    
    subset = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]  # Filter data for the specified year range
    X = subset['Year'].values
    y = subset['Data'].values
    slope, intercept, r_value, p_value, stderr = linregress(X, y)

    z = 1.96  # Z-score for a 95% confidence interval
    ci_lower = slope - z * stderr
    ci_upper = slope + z * stderr
    
    return slope, (ci_lower, ci_upper)

#  Define a function to calculate subsidence rates for each year in a range
def calculate_rates(df, start_year, end_year, window):

    years = np.arange(start_year, end_year + 1)
    results = []
    for year in years:
        # Use a sliding window to calculate the slope
        slope, ci = calculate_slope(df, year - window//2, year + window//2)
        results.append((year, slope, ci[0], ci[1]))
    return pd.DataFrame(results, columns=['year', 'slope', 'ci_lower', 'ci_upper'])

### 1. Calculate the subsidence rate
# Load subsidence data from an Excel file
df = pd.read_excel('treasure_island_dis.xlsx')

dis= df.iloc[:,1]   # Extract subsidence data
years= df.iloc[:,0]    # Extract year data
subsidence = -dis 
# Create a DataFrame to store year and subsidence data
df = pd.DataFrame({'Year': years, 'Data': subsidence})
#  Set parameters
start_year = 1937
end_year = 2080
window = 10  # Sliding window size (e.g., 10 years)

# Save the results
dis_vel = calculate_rates(df, start_year, end_year, window)
dis_vel.to_csv('dis_vel.txt', sep='\t', index=False)

### 2. Calculate the sea level rise rate
df = pd.read_excel('treasure_island_slr.xlsx')

sea_level= df.iloc[:,2]
years= df.iloc[:,0]

df = pd.DataFrame({'Year': years, 'Data': sea_level})
#  Set parameters
start_year = 1939
end_year = 2024
window = 40  

SLR_vel = calculate_rates(df, start_year, end_year, window)
SLR_vel.to_csv('SLR_vel.txt', sep='\t', index=False)