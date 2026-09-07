
These files are part of the paper **"Climate risks of century old land reclamation at Treasure Island".** 

## **Contents** 

**1. InSAR processing**

-opt_phase_est.py

To improve the interferometric phase quality, this python code estimates the optimized phase history from a spatiotemporal homogeneous filtering using homogeneous pixels clustered from their phase similarities. The output interferograms are then used for the PS (Persistent Scatterer) and DS (Distributted Scatterer) processing.

-decompose_LOS_vel.m

This MATLAB code decomposes Line-of-Sight (LOS) velocity measurements from ascending and descending satellite tracks into East–West (EW), North–South (NS), and Vertical (U) components. It incorporates DEM-derived terrain aspect as an additional constraint, assuming that horizontal motion is parallel to the terrain aspect direction. Using the satellite incidence and heading angles, the code calculates the three velocity components for pixels with valid inputs and exports the results as GeoTIFF files. Velocities are expressed in mm/year, with positive LOS values indicating motion toward the satellite.

The script and its input data are stored together in the `3d_deformation/` folder:

| Input files | Description |
| --- | --- |
| `TSX_LOS_A.tif`, `TSX_LOS_D.tif` | Ascending and descending TSX LOS velocities |
| `S1_LOS_A.tif`, `S1_LOS_D.tif` | Ascending and descending Sentinel-1 LOS velocities |
| `aspect.tif` | Terrain aspect derived from the LiDAR DEM |

The script uses TSX data by default. To process Sentinel-1 data, update the input filenames, satellite geometry parameters, and output filenames accordingly.
 

 **2. Deformation prediction from TimeGPT**

 -timegpt_pred_disp.py

This script provides a guide for implementing time series forecasting using TimeGPT with our own data. The code is modified from the official Nixtla TimeGPT repository (https://github.com/Nixtla/nixtla) and Nixtla documentation (https://www.nixtla.io/docs/intro). Before running the prediction, users need to visit dashboard.nixtla.io (https://www.nixtla.io/) to activate a free trial, create a TimeGPT account, and obtain an API key. The Nixtla library should also be installed in the Python environment. 

To reproduce the results, users first prepare the input time series in CSV format, including the observation time and corresponding deformation values. A sample test dataset is provided in the data folder, which can be directly used to verify the code and reproduce the forecasting procedure. The time series is then read and converted into the format required by TimeGPT.

For the code, we first import the Nixtla client and initialize it with the API key. After verifying the validity of the API key, the prepared time series is passed to TimeGPT for forecasting. The forecasting horizon can be specified according to the required number of future time steps. Finally, the predicted results are saved as CSV files for further analysis and visualization. Users can replace the provided test dataset with their own time series data while following the same data preparation and forecasting procedure.

 **3. Preprocessing of inundation analysis data**

### Inverse Distance Weighting interpolation

-IDW_interpolation.py

 This script performs Inverse Distance Weighting (IDW) interpolation to derive study area mean sea level and sea level change rate series from selected tide gauge stations. It reads target grid coordinates, sea level data (from NOAA), IPCC and ITR data (no VLM contribution) from the lat_lon, sea_level, IPCC, and IPCC_vel worksheets in idw.xlsx. The interpolation is applied separately to the monthly mean sea level data, IPCC sea level projections and rate data. The interpolated values are then averaged across all target grid points for each  time step or scenario. These spatially averaged results are exported to text files for further analysis.

### Annual rate calculation

-calculate_annual_rate.py

This script estimates time-varying annual rates of ground subsidence and sea-level rise using centered moving-window linear regression. It reads years and displacement values from treasure_island_dis.xlsx, and years and sea level values from treasure_island_slr.xlsx. For each target year, a linear trend is fitted to the observations within the selected window, and its slope represents the estimated annual rate. Each estimate is accompanied by an approximate 95% confidence interval and standard error. Results are saved to the tab-delimited files dis_vel.txt and SLR_vel.txt, containing the target year, estimated slope, and lower and upper interval bounds. Rate units follow the input measurement units per year.


The script and its input data are stored together in the `inundation_analysis/` folder:
| Input files | Description |
| --- | --- |
| `idw.xlsx` | Target grid coordinates and station-based sea-level observations, projections, and change-rate data for IDW interpolation |
| `treasure_island_dis.xlsx` | Years and displacement values used to estimate annual ground subsidence rates |
| `treasure_island_slr.xlsx` | Years and sea-level values used to estimate annual sea-level rise rates |

**Notes:**

To run the code for new applications, you will need to modify the input data accordingly.

Questions (or if you can’t obtain any file dependency online): Please reach out to Guoqiang SHI at guoqiang.shi@polyu.edu.hk
