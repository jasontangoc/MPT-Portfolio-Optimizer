import pandas as pd
import numpy as np
import yfinance as yf


# def get_tickers():
#     # Tickers are pulled from a maintained CSV file. 
#     # It may become outdated. Wikipedia blocks attempts to scrape.
#     # Will implement a more robust solution later.

#     url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
#     df = pd.read_csv(url)
#     tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
#     return tickers


def get_tickers():
    return ["AAPL", "JNJ", "XOM", "JPM", "NEE"]

def get_historical_data(tickers, time_frame): 
    data = yf.download(tickers, period=time_frame, auto_adjust=True)
    return data["Close"].dropna(axis=1, how="all")



def gen_pct_returns(historical_data):
    pct_returns = historical_data.pct_change().dropna()
    return pct_returns

def gen_log_returns(historical_data):
    log_returns = np.log(historical_data).diff().dropna()
    return log_returns
  




def gen_log_returns_vec(log_returns):
    returns_vector = log_returns.mean() * 252  # Annualize the returns
    return returns_vector


def gen_capm_return_vector(pct_returns, time_frame="1y"):
    # CAPM calculation: R_e = R_f + beta(R_m - R_f)

    risk_free_rate = 0.02  # Example risk-free rate
    market_returns = gen_pct_returns(get_historical_data(["^GSPC"], time_frame))    # S&P 500 as market proxy
    market_rate = (market_returns.mean() * 252).values[0]                           # Annualize market return

    beta = []
    for ticker in pct_returns.columns:
        # beta = cov(R_i, R_m) / var(R_m)
        asset_market_df = pd.DataFrame({
            "Asset Return": pct_returns[ticker],
            "Market Return": market_returns.squeeze()
        }).dropna()
        cov_matrix = asset_market_df.cov()
        beta_value = cov_matrix.loc["Asset Return", "Market Return"] / cov_matrix.loc["Market Return", "Market Return"]
        beta.append(beta_value)
    beta = np.array(beta)
    expected_returns = risk_free_rate + beta * (market_rate - risk_free_rate)
    return expected_returns

def gen_covariance_matrix(returns):
    covariance_matrix = returns.cov() * 252  # Annualize the covariance matrix
    return covariance_matrix

class DataSet:
    def __init__(self, time_frame="1y", CAPM=True):
        self.tickers = get_tickers()
        self.historical_data = get_historical_data(self.tickers, time_frame)
        if CAPM:
            self.returns = gen_pct_returns(self.historical_data)
            self.return_vector = gen_capm_return_vector(self.returns, time_frame)
        else:
            self.returns = gen_log_returns(self.historical_data)
            self.return_vector = gen_log_returns_vec(self.returns)
        self.covariance_matrix = gen_covariance_matrix(self.returns)




