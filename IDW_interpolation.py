# This script performs Inverse Distance Weighting (IDW) interpolation

"""
Note - 
"""

import numpy as np
import pandas as pd
from math import radians, cos, sin, sqrt, atan2

#  Define a function to calculate the spherical distance between two lat/lon points
def haversine(lon1, lat1, lon2, lat2):
    # Convert latitude and longitude to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Define Earth's radius (in kilometers)
    R = 6372.8
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Calculate distance
    return R * c

 # Define a function to Perform IDW interpolation
def idw_interpolation(lat, lon, p=2):
    # Calculate distances and weights
    distances = {station: haversine(lon, lat, data['lon'], data['lat']) for station, data in stations.items()}
    weights = {station: 1 / (dist**p) for station, dist in distances.items()}
    sum_of_weights = sum(weights.values())

    # Interpolate 
    interp_data = []
    for i in range(len(SF)):
        weighted_sum = sum(stations[station]['data'][i] * weights[station] for station in stations)
        interp_data.append(weighted_sum / sum_of_weights)

    return np.array(interp_data)


# Define known stations with their latitude, longitude
stations = {
    'SF': {'lat': 37.806667, 'lon': -122.465 },
    'AL': {'lat': 37.771667, 'lon': -122.298333 }
}

# Read grid points (latitude and longitude) from Excel file
lat_lon = pd.read_excel('idw.xlsx', sheet_name='lat_lon')
interp_grid_points = list(zip(lat_lon.iloc[:, 0], lat_lon.iloc[:, 1]))

# 1. Generate a time series of monthly mean sea level data

# Read data from Excel file
df = pd.read_excel('idw.xlsx', sheet_name='sea_level')
# Extract sea level data for stations
SF = df.iloc[:, 0]
AL = df.iloc[:, 1]
stations['SF']['data'] = SF
stations['AL']['data'] = AL

# Interpolate sea levels for each grid point
interp_grid = np.array([idw_interpolation(lat, lon) for lat, lon in interp_grid_points])

# Calculate the average sea level for each time point across all grid points
mean_data = np.mean(interp_grid, axis=0)

# Export the average sea level results to a text file
output_file = "sea_level.txt"
with open(output_file, 'w') as f:
    f.write("Sea Level (m)\n")
    f.writelines(f"{sea_level:.4f}\n" for sea_level in mean_data)

print(f"The average sea level data for the interpolated grid points has been saved to {output_file}")


# 2. Generate a time series of IPCC data
df = pd.read_excel('idw.xlsx', sheet_name='IPCC')

SF= df.iloc[:, :14].values.reshape(-1,)
AL= df.iloc[:, 15:].values.reshape(-1,)
stations['SF']['data'] = SF
stations['AL']['data'] = AL

# Interpolate sea levels for each grid point
interp_grid = np.array([idw_interpolation(lat, lon) for lat, lon in interp_grid_points])

# Calculate the average sea level for each time point across all grid points
mean_data = np.mean(interp_grid, axis=0)
# Export the IPCC sea level results to a text file
output_file = "IPCC.txt"
with open(output_file, 'w') as f:
    f.write("IPCC (m)\n")
    f.writelines(f"{sea_level:.4f}\n" for sea_level in mean_data)

print(f"The IPCC data for the interpolated grid points has been saved to {output_file}")

# 3. Generate the change rate of IPCC data
df = pd.read_excel('idw.xlsx', sheet_name='IPCC_vel')

SF= df.iloc[:, :14].values.reshape(-1,)
AL= df.iloc[:, 15:].values.reshape(-1,)
stations['SF']['data'] = SF
stations['AL']['data'] = AL

# Interpolate sea levels for each grid point
interp_grid = np.array([idw_interpolation(lat, lon) for lat, lon in interp_grid_points])

# Calculate the average sea level for each time point across all grid points
mean_data = np.mean(interp_grid, axis=0)
# Export the IPCC sea level results to a text file
output_file = "IPCC_vel.txt"
with open(output_file, 'w') as f:
    f.write("IPCC_vel (m)\n")
    f.writelines(f"{sea_level:.4f}\n" for sea_level in mean_data)

print(f"The IPCC_vel data for the interpolated grid points has been saved to {output_file}")

