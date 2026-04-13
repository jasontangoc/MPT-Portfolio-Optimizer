from data_retrieval import *
from monte_carlo import *
from optimizer import *

def main():
    df = DataSet()
    # fig, portfolio_data = run_monte_carlo(df.tickers, df.return_vector, df.covariance_matrix)
    # fig.show()

    weights = optimize_portfolio(df)
    for ticker, weight in zip(df.tickers, weights):
        print(f"{ticker}: {weight:.2%}")
    print(sum(weights))

if __name__ == "__main__":
    main()
    