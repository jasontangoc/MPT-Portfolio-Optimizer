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
     
def generate_log_returns(historical_data):
    log_returns = np.log(historical_data).diff()
    return log_returns.dropna()

def generate_return_vector(log_returns):
    # R_e = R_f + beta(R_m - R_f), will implement later
    return_vector = log_returns.mean() * 252  # Annualize the returns
    return return_vector

def generate_covariance_matrix(log_returns):
    covariance_matrix = log_returns.cov() * 252  # Annualize the covariance matrix
    return covariance_matrix

class DataSet:
    def __init__(self, time_frame="1y"):
        self.tickers = get_tickers()
        self.historical_data = get_historical_data(self.tickers, time_frame)
        self.log_returns = generate_log_returns(self.historical_data)
        self.return_vector = generate_return_vector(self.log_returns)
        self.covariance_matrix = generate_covariance_matrix(self.log_returns)




