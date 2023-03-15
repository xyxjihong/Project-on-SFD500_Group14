#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:40:43 2023

@author: Xu Yaxuan
"""
#%% 1. packages import
import numpy as np
import pandas as pd
import math
import datetime as dt
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import yfinance as yf
import value_SFD500 as cal
import structure_product
import warnings
warnings.filterwarnings("ignore")
import importlib
importlib.reload(structure_product)

#%% 2. MC init settings
# import trading days
date = pd.to_datetime(pd.read_csv('Data\Project2 business_day.csv')['date'])
## assert constant
startDate, endDate = dt.datetime(2019,1,4), dt.datetime(2019,12,27)
# monitor dates for up-and-out barrier
UnO_dates = pd.to_datetime(pd.Series([dt.date(2019,2,1)
             ,dt.date(2019,3,4)
             ,dt.date(2019,3,29)
             ,dt.date(2019,4,26)
             ,dt.date(2019,5,31)
             ,dt.date(2019,6,28)
             ,dt.date(2019,7,26)
             ,dt.date(2019,8,23)
             ,dt.date(2019,9,20)
             ,dt.date(2019,10,25)
             ,dt.date(2019,11,22)
             ,dt.date(2019,12,27)]))
# mutual paras
face_value = 100
S0 = 5.10
R = 0.28
h_u = 1.05*S0
h_d = 0.68*S0

#%% 3. calculate simulated return and suvival time
price = pd.read_csv("Data\Project2 SFD500-stock_price-002727.csv")
price.replace(0, np.nan, inplace=True)
sigma = math.sqrt(np.mean((np.log(price['CLOSE']/price['CLOSE'].shift(1)))**2)*252)
r = math.log(1+0.015)
q = 0.0127009829447111

#%% 4. MC value and se
MU = []
SE = []
for i in range(1000,101000,1000):
    n = i
    mu, se, survival_time_avg, result = cal.value_SFD500(S0, r, q, sigma, R, n, date, UnO_dates ,h_u, h_d, face_value)
    MU.append(mu); SE.append(se)
fig, ax = plt.subplots()
ax.plot(np.arange(1000,101000,1000), np.array(MU),'b-', label = 'Product Value')
ax2 = ax.twinx()
ax2.plot(np.arange(1000,101000,1000), np.array(SE), 'r-', label = 'Standard Error')
ax.legend(bbox_to_anchor=(0.99, 1))
ax2.legend(bbox_to_anchor=(1, 0.95))
ax.set_xlabel("# of Sample Path")
ax.set_ylabel("Product Value")
ax2.set_ylabel("Standard Error")
ax.axhline(y = 96.3, lw = 1.5, ls = "--",c = "grey")
plt.savefig("result.png")

#%% 5. Hedging Strat


#%% 6. MC return
n = 100000
mu, se, survival_time_avg, result = cal.value_SFD500(S0, r, q, sigma, R, n, date, UnO_dates ,h_u, h_d, face_value)
# return histogram
result.hist(column='return', bins=15, grid=False, figsize=(10,6),rwidth = 0.9)

#%% 7. Real stock price survival time and return
real_path = yf.download("002027.SZ", start="2019-01-03", end="2019-12-28")['Close']
knock_out_flag = (real_path[UnO_dates]>=h_u).any()
knock_in_flag = (real_path<h_d).any()
knockdate = (real_path[UnO_dates]>=h_u).idxmax()
survival_time_index = int(date[date == knockdate].index.values) 
survival_period = (date[survival_time_index] - date[1]).days * knock_out_flag
payoff = face_value*(1+R*survival_period/365)*knock_out_flag +\
                    face_value*np.minimum(real_path[endDate]/5.1,1)*knock_in_flag*(1-knock_out_flag) +\
                    face_value*(1+R*357/365)*(1-knock_in_flag)*(1-knock_out_flag)
real_return = payoff / mu -1
estimated_return = [result["return"].mean(),result["return"].median(),result["return"].std()]
estimated_survival = [result["survival time"].mean(),result["survival time"].median(),result["survival time"].std()]

#%% Summary
print("summary for Q1 MC")
print("value = "+str(mu))
print("se = " + str(se))
print("summary for Q3")
print("estimated return (mean,med,std): " + str(estimated_return))
print("estimated survival (mean,med,std): " + str(estimated_survival))
print("real return = " + str(real_return))
print("real survival = " + str(survival_period/365))





