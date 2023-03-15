import numpy as np
import pandas as pd
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

class SFD500():

    def __init__(self, start_date, end_date, face_value, S0, UnO_factor, DnI_factor, UnO_dates, DnI_dates, R, strike, trade_dates):
        self.start_date = start_date # start date 
        self.end_date = end_date # end date
        self.face_value = face_value # face value
        self.S0 = S0 # initial stock price
        self.UnO_barrier = UnO_factor * S0 # up-and-out barrier level 
        self.DnI_barrier = DnI_factor * S0 # down-and-in barrier level 
        self.UnO_dates = UnO_dates # up-and-out barrier monitoring dates
        self.DnI_dates = DnI_dates # down-and-in barrier monitoring dates
        self.R = R # annualized coupon rate
        self.strike = strike # strike price
        self.trade_dates = trade_dates # trading dates
        self.T = (end_date - start_date).days # product period in days
        self.expiry_payoff_date = (trade_dates[trade_dates > end_date]).iloc[0] # payment date after expiry

    def get_mc_price(self, value_date, s0, r, q, sigma, n, knock_in):
        m = int(self.trade_dates[self.trade_dates == self.end_date].index.values) - int(self.trade_dates[self.trade_dates == value_date].index.values) # number of trading days to expiry
        s = np.zeros([n, m+1], dtype = np.float64)
        s[:, 0] = s0
        z = np.random.randn(n, m)
        delta_t = 1/365
        for i in range(m): # generate n samples of stock price at each time step
            s[:, i+1] = s[:, i] * np.exp((r - q - 0.5*sigma**2)*delta_t + sigma*np.sqrt(delta_t)*z[:,i])
        price_path = pd.DataFrame(s, columns = self.trade_dates[int(self.trade_dates[self.trade_dates == value_date].index.values) : int(self.trade_dates[self.trade_dates == self.end_date].index.values)+1])
        
        # knock-out event
        knock_out_flag = (price_path[self.UnO_dates[self.UnO_dates.isin(price_path.columns)]] >= self.UnO_barrier).any(axis = 1) # indicator of up-and-out paths
        knockdate_list = (price_path[self.UnO_dates[self.UnO_dates.isin(price_path.columns)]] >= self.UnO_barrier).idxmax(axis = 1) # series of knock-out dates
        survival_time_index = [int(self.trade_dates[self.trade_dates == knockdate].index.values) for knockdate in knockdate_list]
        survival_period = np.array([(self.trade_dates[j] - self.start_date).days for j in survival_time_index]) 
        # survival time
        survival_time = (knock_out_flag * survival_period + (1-knock_out_flag) * self.T)/365
        # knock-out payoff
        payoff_time = [self.trade_dates[j] for j in (np.array(survival_time_index) + 1).tolist()] # payment dates after up-and-out events
        discount_period = np.array([(payoff_time_i - value_date).days for payoff_time_i in payoff_time]) # number of days to payment dates 
        knock_out_payoff = self.face_value * (1 + self.R * survival_period/365) * np.exp(-r * discount_period/365) * knock_out_flag # current value of up-and-out paths
        
        # knock-in event
        knock_in_flag = pd.Series([True] * n) if knock_in else (price_path[self.DnI_dates[self.DnI_dates.isin(price_path.columns)]] < self.DnI_barrier).any(axis = 1) # indicator of down-and-in paths
        # expiration payoff time 
        expiry_payoff_period = (self.expiry_payoff_date - value_date).days/365 # number of days to payment dates 
        # knock-in payoff
        knock_in_payoff = self.face_value * np.minimum(price_path[self.end_date]/self.S0, self.strike) * np.exp(-r * expiry_payoff_period) * knock_in_flag * (1-knock_out_flag)
        # non-knock payoff
        non_knock_payoff = self.face_value * (1 + self.R * 357/365) * np.exp(-r * expiry_payoff_period) * (1-knock_in_flag) * (1-knock_out_flag)

        # result
        result = pd.DataFrame(index = price_path.index)
        result["value"] = knock_out_payoff + knock_in_payoff + non_knock_payoff
        result["survival time"] = survival_time
        result["knock in flag"] = knock_in_flag
        result["knock out flag"] = knock_out_flag
        result["non knock"] = (1-knock_in_flag) * (1-knock_out_flag)
        mu = result["value"].mean()
        se = np.sqrt(((result["value"]**2).sum() - n*mu**2)/n/(n-1))
        survival_time_avg = result["survival time"].mean()
        return mu, se, survival_time_avg, result

    def get_mc_price_daily(self, r, q, sigma, n, v0):
        m = int(self.trade_dates[self.trade_dates == self.end_date].index.values) - int(self.trade_dates[self.trade_dates == self.start_date].index.values) + 1 # number of trading days to expiry
        s = np.zeros([n, m+1], dtype = np.float64) # simulated stock price
        v = np.zeros([n, m+1], dtype = np.float64) # estimated product value
        s[:, 0] = self.S0 # initial stock price
        v[:, 0] = v0 # initial product value
        z = np.random.randn(n, m)
        delta_t = 1/365
        for i in range(m): # generate n samples of stock price at each time step
            s[:, i+1] = s[:, i] * np.exp((r - q - 0.5*sigma**2)*delta_t + sigma*np.sqrt(delta_t)*z[:,i])
        price_path = pd.DataFrame(s, columns = self.trade_dates[int(self.trade_dates[self.trade_dates == self.start_date].index.values)-1 : int(self.trade_dates[self.trade_dates == self.end_date].index.values)+1])
        
        # knock-out event
        knock_out_flag = (price_path[self.UnO_dates] >= self.UnO_barrier).any(axis = 1) # indicator of up-and-out paths
        knock_out_dates = (price_path[self.UnO_dates] >= self.UnO_barrier).idxmax(axis = 1) # series of knock-out dates
        survival_time_index = [int(self.trade_dates[self.trade_dates == knockdate].index.values) for knockdate in knock_out_dates]
        survival_period = np.array([(self.trade_dates[j] - self.start_date).days for j in survival_time_index]) 
        # knock-out payoff
        payoff_time = pd.Series([self.trade_dates[j] for j in (np.array(survival_time_index) + 1).tolist()]) # payment dates after up-and-out events
        knock_out_payoff = self.face_value * (1 + self.R * survival_period/365) * knock_out_flag # value of up-and-out paths on payment dates
        knock_out_date_value = knock_out_payoff * np.exp(-r * (payoff_time - knock_out_dates).dt.days/365) # value of up-and-out paths on knock-out dates
        
        # knock-in event
        knock_in_flag = (price_path[self.DnI_dates] < self.DnI_barrier).any(axis = 1) # indicator of down-and-in paths
        knock_in_dates = (price_path[self.DnI_dates] < self.DnI_barrier).idxmax(axis = 1) # series of knock-in dates

        # estimate product price on each trading date
        for i in range(1, m+1):
            value_date = price_path.columns.values[i] # current valuation date
            knock_out_dates_index = np.where(knock_out_flag & (knock_out_dates == value_date))[0] # paths on knock-out date
            payoff_dates_index = np.where(knock_out_flag & (payoff_time == value_date))[0] # paths on payment dates after knock-out events
            if np.any(knock_out_dates_index):
                v[knock_out_dates_index, i] = knock_out_date_value[knock_out_dates_index] 
            if np.any(payoff_dates_index):
                v[payoff_dates_index, i] = knock_out_payoff[payoff_dates_index]
            for j in np.argwhere((~knock_out_flag.to_numpy()) | (knock_out_dates.to_numpy() > value_date)): # paths with no knock-out event or before knock-out events
                v[j[0], i] = self.get_mc_price(value_date, s[j[0], i], r, q, sigma, n, (knock_in_flag[j[0]] and (value_date >= knock_in_dates[j[0]])))[0]
        result = pd.DataFrame(v, columns = price_path.columns)
        return result
    
    def get_real_price_daily(self, s, r, q, sigma, n, v0):
        v = np.zeros(s.size) # estimated product value
        v[0] = v0 # initial product value
        
        # knock-out event
        knock_out_flag = (s[self.UnO_dates] >= self.UnO_barrier).any() # indicator of up-and-out event
        knock_out_date = (s[self.UnO_dates] >= self.UnO_barrier).idxmax() # knock-out date
        survival_time_index = int(self.trade_dates[self.trade_dates == knock_out_date].index.values) 
        survival_period = (self.trade_dates[survival_time_index] - self.start_date).days 
        # knock-out payoff
        payoff_time = self.trade_dates[survival_time_index + 1] # payment dates after the up-and-out event
        knock_out_payoff = self.face_value * (1 + self.R * survival_period/365) * knock_out_flag # up-and-out value on payment date
        knock_out_date_value = knock_out_payoff * np.exp(-r * (payoff_time - knock_out_date).days/365) # up-and-out value on knock-out date
        
        # knock-in event
        knock_in_flag = (s[self.DnI_dates] < self.DnI_barrier).any() # indicator of down-and-in event
        knock_in_date = (s[self.DnI_dates] < self.DnI_barrier).idxmax() # knock-in date

        # estimate product price on each trading date
        for i in range(1, s.size):
            value_date = s.index.values[i] # current valuation date
            if knock_out_flag and (knock_out_date == value_date):
                v[i] = knock_out_date_value
            elif knock_out_flag and (payoff_time == value_date):
                v[i] = knock_out_payoff
            elif (~knock_out_flag) or (knock_out_date > value_date):
                v[i] = self.get_mc_price(value_date, s[i], r, q, sigma, n, (knock_in_flag and (value_date >= knock_in_date)))[0]
        v = pd.Series(v, index = s.index)
        return v

    def set_UnO_cdf(self, value_date, s0, r, q, sigma, ds):
        T = (self.end_date - value_date).days/365 # current time to expiry in year
        n_UnO_dates = (self.UnO_dates > value_date).sum() # remaining number of up-and-out barrier monitoring dates 
        t = np.array([*range(n_UnO_dates+1)]) / n_UnO_dates 
        drift = (r - q - 0.5*sigma**2) * np.sqrt(T) / sigma 
        def cdf(s):
            barrier = np.log(self.UnO_barrier/s) / sigma / np.sqrt(T)
            G = 1 - norm.cdf(barrier/np.sqrt(t) - drift*np.sqrt(t)) + np.exp(2*drift*barrier) * norm.cdf(-barrier/np.sqrt(t) - drift*np.sqrt(t)) # cdf of t
            return G
        self.UnO_cdf = [cdf(s0 + ds), cdf(s0 - ds)]

    def get_delta(self, value_date, s0, r, q, sigma, n, ds):
        if (pd.Timestamp(value_date) in self.UnO_dates.values) or value_date < self.start_date: # currently on a up-and-out barrier monitoring date or before the start date
            self.set_UnO_cdf(value_date, s0, r, q, sigma, ds) # reset cdf for remaining up-and-out barrier monitoring dates
        m = int(self.trade_dates[self.trade_dates == self.end_date].index.values) - int(self.trade_dates[self.trade_dates == value_date].index.values) # number of trading days to expiry
        S = np.zeros([n, m+1], dtype = np.float64)
        # non-knock payoff
        expiry_payoff_period = (self.expiry_payoff_date - value_date).days/365 # number of days to payment dates
        non_knock_payoff = self.face_value * (1 + self.R * 357/365) * np.exp(-r * expiry_payoff_period)
        # knock-out payoff     
        knock_out_dates = self.UnO_dates[self.UnO_dates > value_date] # remaining up-and-out barrier monitoring dates
        survival_period = (knock_out_dates - self.start_date).dt.days # survival time in days 
        knock_out_dates_index = [int(self.trade_dates[self.trade_dates == knockdate].index.values) for knockdate in knock_out_dates]
        payoff_dates = pd.Series([self.trade_dates[j] for j in (np.array(knock_out_dates_index) + 1).tolist()]) # payment dates after up-and-out events
        discount_period = (payoff_dates - value_date).dt.days # number of days to payment dates 
        knock_out_payoff = self.face_value * (1 + self.R * survival_period/365) * np.exp(-r * discount_period/365) # current value of up-and-out payoff 
        def non_knock_out_value(s):
            S[:, 0] = s
            z = np.random.randn(n, m)
            delta_t = 1/365
            for i in range(m): # generate n samples of stock price at each time step
                S[:, i+1] = S[:, i] * np.exp((r - q - 0.5*sigma**2)*delta_t + sigma*np.sqrt(delta_t)*z[:,i])
            price_path = pd.DataFrame(S, columns = self.trade_dates[int(self.trade_dates[self.trade_dates == value_date].index.values) : int(self.trade_dates[self.trade_dates == self.end_date].index.values)+1])
            # knock-in event
            knock_in_flag = (price_path[self.DnI_dates[self.DnI_dates.isin(price_path.columns)]] < self.DnI_barrier).any(axis = 1) # indicator of down-and-in paths
            # knock-in payoff
            knock_in_payoff = self.face_value * np.minimum(price_path[self.end_date]/self.S0, self.strike) * np.exp(-r * expiry_payoff_period)
            # non-knock-out payoff
            non_knock_out_payoff = knock_in_payoff * knock_in_flag + non_knock_payoff * (1 - knock_in_flag)
            return np.mean(non_knock_out_payoff) 
        v0 = non_knock_out_value(s0 + ds) * (1 - self.UnO_cdf[0][-1]) + np.sum(knock_out_payoff * np.diff(self.UnO_cdf[0])) * self.UnO_cdf[0][-1]
        v1 = non_knock_out_value(s0 - ds) * (1 - self.UnO_cdf[1][-1]) + np.sum(knock_out_payoff * np.diff(self.UnO_cdf[1])) * self.UnO_cdf[1][-1]
        return (v0 - v1) / (2*ds)

    def test_delta_hedge(self, stock_series, value_series, r, q, sigma, n, ds):
        delta_series = []
        for date in stock_series.index: # compute delta on every trading date before the knock-out event if any
            if (pd.Timestamp(date) in self.UnO_dates.values) and (stock_series[date] >= self.UnO_barrier):
                delta_series.append(0)
                break
            else:
                delta_series.append(self.get_delta(date, stock_series[date], r, q, sigma, n, ds))
        delta_series = pd.Series(delta_series, index = stock_series.index[:len(delta_series)]).rename("delta")
        dates = delta_series.index # dates during the real survival period
        dt = pd.Series(dates, index = dates).diff().dt.days / 365 # lags between trading dates in year
        bond = (value_series[dates] - delta_series * stock_series[dates]).rename("bond") # positions in the risk-free asset
        pnl = (-value_series[dates].diff() + delta_series.shift(1) * stock_series[dates].diff() + (np.exp(r*dt) - 1) * bond.shift(1)).rename("PnL") # daily profit and loss
        cum_pnl = pnl.cumsum().rename("cumulative PnL") # cumulative profit and loss 
        return pd.concat([delta_series, bond, pnl, cum_pnl], axis = 1)

    def plot_real_mc_price(self, real, mc_mean, mc_median, color_real = "blue", color_mean = "green", color_median = "orange", x_label = "Date", y_label = "Price", 
                           title = "Estimated price using Monte-Carlo Simulation", width = 800, height = 600):
    
        fig = make_subplots(rows = 1, cols = 1)
        # plot real price
        fig.add_trace(go.Scatter(x = real.index, y = real, line = dict(color = color_real, width = 1), name = "Real"), row = 1, col = 1)
        # plot mean of Monte-Carlo estimates
        fig.add_trace(go.Scatter(x = mc_mean.index, y = mc_mean, line = dict(color = color_mean, width = 1), name = "MC Mean"), row = 1, col = 1)
        # plot median of Monte-Carlo estimates
        fig.add_trace(go.Scatter(x = mc_median.index, y = mc_median, line = dict(color = color_median, width = 1), name = "MC Median"), row = 1, col = 1)
        fig.update_xaxes(
            title = {"text": x_label},
            linecolor = "black",
            mirror = True)
        fig.update_yaxes(
            title = {"text": y_label},
            linecolor = "black",
            mirror = True)
        fig.update_layout(
            title = {"text": title, "x": 0.5},
            autosize = False,
            width = width,
            height = height,
            plot_bgcolor = "rgba(0, 0, 0, 0)",
            legend = dict(yanchor = "top", y = 1, xanchor = "right", x = 1)
        )
        fig.write_html("price_series.html")
        fig.show()

    def plot_delta_pnl(self, delta_series, pnl_series, cum_pnl_series, color_delta = "blue", color_pnl = "green", color_cum_pnl = "orange", color_mean = "grey", x_label = "Date", 
                       title = "Estimated delta and daily profit and loss over the product survival period", width = 1000, height = 500, linewidth = 1):
    
        fig = make_subplots(rows = 1, cols = 2)
        # plot delta
        # fig.add_trace(go.Bar(x = delta_series.index, y = delta_series, marker_color = color_delta, name = "Delta"), row = 1, col = 1)
        fig.add_trace(go.Scatter(x = delta_series.index, y = delta_series, line = dict(color = color_delta, width = linewidth), name = "Delta"), row = 1, col = 1)
        fig.add_shape(type = "line", x0 = delta_series.index[0], y0 = delta_series.mean(), x1 = delta_series.index[-1], y1 = delta_series.mean(),
                      line = dict(color = color_mean, dash = "dash"), xref = "x", yref = "y", row = 1, col = 1)
        # plot daily PnL
        fig.add_trace(go.Scatter(x = pnl_series.index, y = pnl_series, line = dict(color = color_pnl, width = linewidth), name = "Daily PnL"), row = 1, col = 2)
        fig.add_trace(go.Scatter(x = cum_pnl_series.index, y = cum_pnl_series, line = dict(color = color_cum_pnl, width = linewidth), name = "Cumulative PnL"), row = 1, col = 2)
        fig.add_shape(type = "line", x0 = pnl_series.index[0], y0 = pnl_series.mean(), x1 = pnl_series.index[-1], y1 = pnl_series.mean(),
                      line = dict(color = color_mean, dash = "dash"), xref = "x", yref = "y", row = 1, col = 2)
        fig.update_xaxes(
            title = {"text": x_label},
            linecolor = "black",
            mirror = True)
        fig.update_yaxes(
            linecolor = "black",
            mirror = True)
        fig.update_layout(
            title = {"text": title, "x": 0.5},
            autosize = False,
            width = width,
            height = height,
            plot_bgcolor = "rgba(0, 0, 0, 0)",
            legend = dict(yanchor = "bottom", y = 0, xanchor = "right", x = 1),
            yaxis2 = dict(range = [-20, 20])
        )
        fig.write_html("delta_pnl_series.html")
        fig.show()

def get_dividend_yield(price_series: pd.Series, dividend_series: pd.Series) -> float:
    ''' 
    Return estimated continuously compounded dividend yield accoridng to the average annual dividend yield 
    '''

    price_dividend = pd.concat([price_series, dividend_series], axis = 1)
    annual_yield = (price_dividend[price_dividend.columns[1]]/price_dividend[price_dividend.columns[0]].shift(1)).mean() # average annual dividend yield
    return np.log(1 + annual_yield)

def get_max_drawdown(cum_pnl_series: pd.Series) -> float:
    trough_id = (cum_pnl_series.cummax() - cum_pnl_series).idxmax()
    return ((cum_pnl_series.cummax() - cum_pnl_series)/cum_pnl_series.cummax())[trough_id] * 100

def get_annual_sharpe(pnl_series: pd.Series) -> float:
    return np.sqrt(252) * pnl_series.mean() / pnl_series.std()