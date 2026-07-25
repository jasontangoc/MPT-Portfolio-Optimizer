import os

from data_retrieval import *
from monte_carlo import *
from optimizer import *

def main():
    df = DataSet()

    fig, portfolio_data = run_monte_carlo(df.tickers, df.return_vector, df.covariance_matrix)
    os.makedirs("images", exist_ok=True)
    fig.write_image("images/efficient_frontier.png", width=900, height=600)

    weights = optimize_portfolio(df)
    for ticker, weight in zip(df.tickers, weights):
        print(f"{ticker}: {weight:.2%}")

if __name__ == "__main__":
    main()
