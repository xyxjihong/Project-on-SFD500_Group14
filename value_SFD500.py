import numpy as np
import pandas as pd
import math
import datetime as dt

def value_SFD500(S0, r, q, sigma, R, n, date, UnO_dates ,h_u, h_d, face_value):
    startDate, endDate = dt.datetime(2019,1,4), dt.datetime(2019,12,27)
    T = (endDate-startDate).days
    m = int(date[date == endDate].index.values) - int(date[date == startDate].index.values) + 1
    s = np.zeros([n,m+1], dtype = np.float64)
    s[:,0] = S0
    z = np.random.randn(n,m)
    delta_t = 1/365
    for i in range(m):
        s[:,i+1] = s[:,i]*np.exp((r - q - 0.5*sigma**2)*delta_t+sigma*math.sqrt(delta_t)*z[:,i])
    price_path = pd.DataFrame(s,columns = date[int(date[date == startDate].index.values)-1: int(date[date == endDate].index.values) +1])
    # knock_out_event
    knock_out_flag = (price_path[UnO_dates]>=h_u).any(axis = 1)
    knockdate_list = (price_path[UnO_dates]>=h_u).idxmax(axis = 1)
    survival_time_index = [int(date[date == knockdate].index.values) for knockdate in knockdate_list]
    survival_period = np.array([(date[j] - date[1]).days for j in survival_time_index])
    # survival_time
    survival_time = (knock_out_flag*survival_period +(1-knock_out_flag)*T)/365
    # knock_out_discounted_payoff
    payoff_time = [date[j] for j in (np.array(survival_time_index) + 1).tolist()]
    discount_period = np.array([(payoff_time_i-date[0]).days for payoff_time_i in payoff_time])
    knock_out_payoff = face_value*(1+R*survival_period/365)*np.exp(-r*discount_period/365)*knock_out_flag
    # knock_in_event
    knock_in_flag = (price_path<h_d).any(axis = 1)
    # expiration_payoff_time 
    expiration_payoff_date = (date[date > endDate]).iloc[0]
    expiration_payoff_period = (expiration_payoff_date - date[0]).days/365
    # knock_in_discounted_payoff
    knock_in_payoff = face_value*np.minimum(price_path[endDate]/5.1,1)*math.exp(-r*expiration_payoff_period)*knock_in_flag*(1-knock_out_flag)
    # non_knock_discounted_payoff
    non_knock_payoff = face_value*(1+R*357/365)*math.exp(-r*expiration_payoff_period)*(1-knock_in_flag)*(1-knock_out_flag)
    # result
    result = pd.DataFrame(index = price_path.index)
    result["payoff"] = face_value*(1+R*survival_period/365)*knock_out_flag +\
                        face_value*np.minimum(price_path[endDate]/5.1,1)*knock_in_flag*(1-knock_out_flag) +\
                        face_value*(1+R*357/365)*(1-knock_in_flag)*(1-knock_out_flag)
    result["value"] = knock_out_payoff + knock_in_payoff + non_knock_payoff
    result["survival time"] = survival_time
    result["return"] = result["payoff"]/result["value"]-1
    result["knock in flag"] = knock_in_flag
    result["knock out flag"] = knock_out_flag
    result["non knock"] = (1-knock_in_flag)*(1-knock_out_flag)
    mu = result["value"].mean()
    se = np.sqrt(((result["value"]**2).sum() - n*mu**2)/n/(n-1))
    survival_time_avg = result["survival time"].mean()
    return mu, se, survival_time_avg, result
    