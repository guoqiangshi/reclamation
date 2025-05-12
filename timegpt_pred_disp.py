#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 19:18:59 2024

@author: shgq
"""

from nixtla import NixtlaClient
nixtla_client = NixtlaClient(
    api_key = 'nixtla-tok-mUf337YTELsmzyFM6KEuwKNzJtp0NgJiSsXw5NM2gZ8HSqlBGAXsGKsFOI2MxUhhUkqpoRp6N3tlgqOa'
)
nixtla_client.validate_api_key()

#####---------------------------- test -----------------------------------
import pandas as pd
import matplotlib.pyplot as plt


## data input
import os
path='/home/shgq/Disk1/treasure_island/data/TS'
files=os.listdir(path)
files.sort(key=lambda x: int(x.split(".")[0].split("_")[1]))
print(files)

## forecast
files_csv=[f for f in files if f[-4:] == '.csv']
df_pred=[]
for file in files_csv:
    df=pd.read_csv(os.path.join(path,file))
    pp0=nixtla_client.forecast(df=df, h=60,finetune_steps=20,model='timegpt-1-long-horizon', time_col='date', target_col='def')
    df_pred.append(pp0)
# model='timegpt-1-long-horizon',

## plot one prediction
fig=plt.figure()
timegpt_fcst_df=df_pred[19]
plt.plot(timegpt_fcst_df['date'],timegpt_fcst_df['TimeGPT'])   

## save predicted results
preds0=pd.DataFrame()
for i in range(len(df_pred)):
    d0=df_pred[i]['TimeGPT']
    preds0[i]=d0
date0=df_pred[0]['date']
preds_0=pd.concat([date0,preds0],axis=1)
preds_0.to_csv("./data/results0/preds_z_80.csv")    
    

