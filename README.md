
These files are part of the paper **"Century-long Evolution of Reclaimed Land under Climate Change".** 

## **Contents** 

**1. InSAR processing**

-opt_phase_est.py

To improve the interferometric phase quality, this python code estimates the optimized phase history from a spatiotemporal homogeneous filtering using homogeneous pixels clustered from their phase similarities. The output interferograms are then used for the PS (Persistent Scatterer) and DS (Distributted Scatterer) processing.

-decomposeLOS_vel.m

This MATLAB code decomposes Line-of-Sight (LOS) velocity measurements from ascending and descending satellite tracks into their East-West (EW), North-South (NS), and Vertical (U) components. 

 **2. Deformation prediction from TimeGPT**

 -timegpt_pred_disp.py

This script provides a guide for implementing time series forecasting using TimeGPT with our own data. The code is modified from https://github.com/Nixtla/nixtla and https://www.nixtla.io/docs/intro. Before executing the code for prediction, we need some preparatory work. We need visit [dashboard.nixtla.io](https://dashboard.nixtla.io/) to activate free trial and create a TimeGPT account. New API key is necessary for the following prediction and Nixtla library must be installed in our Python environment. 

For the code, we firstly import the Nixtla client, instantiate it with our API key and verify the status and validity of API key. After ensuring that API is valid, we then input the preprocessed time series into TimeGPT. Finally, we save all the prediction results to csv files.

 **3. Preprocessing of inundation analysis data**

-IDW_interpolation.py

 This script performs Inverse Distance Weighting (IDW) interpolation. The script reads input data from an Excel file containing latitude/longitude grid points, sea level data (from NOAA), IPCC and ITR data (no VLM contribution). The results are exported to text files for further analysis.

-calculate_annual_rate.py

This script calculates the annual change rate of ground subsidence and sea level rise. The results are saved to text files and can be used to analyze trends over time. The script reads input data from  Excel files containing vertical land subsidence data and sea level data. 

**Notes:**

To run the code for new applications, you will need to modify the input data accordingly.

Questions (or if you can’t obtain any file dependency online): Please reach out to Guoqiang SHI at guoqiang.shi@polyu.edu.hk
