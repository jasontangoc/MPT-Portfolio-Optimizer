import numpy as np
import pandas as pd
import plotly.express as ex
from data_retrieval import DataSet

def calculate_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.0):
    port_return = np.dot(weights, expected_returns)
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_volatility = np.sqrt(port_variance)
    return (port_return - risk_free_rate) / port_volatility

def run_monte_carlo(tickers, expected_returns, cov_matrix, num_portfolios=10000):
    num_assets = len(tickers)

    results_returns = []
    results_volatility = []
    results_sharpes = []
    results_weights = []

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)

        port_return = np.dot(weights, expected_returns)
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_volatility = np.sqrt(port_variance)
        sharpe = calculate_sharpe_ratio(weights, expected_returns, cov_matrix)

        results_returns.append(port_return)
        results_volatility.append(port_volatility)
        results_sharpes.append(sharpe)
        results_weights.append(weights)

    portfolio_data = pd.DataFrame({
        "Return": results_returns,
        "Volatility": results_volatility,
        "Sharpe Ratio": results_sharpes
    })

    for counter, ticker in enumerate(tickers):
        portfolio_data[f"{ticker} Weight"] = [w[counter] for w in results_weights]

    fig = ex.scatter(
        portfolio_data,
        x="Volatility",
        y="Return",
        color="Sharpe Ratio",
        color_continuous_scale="Viridis",
        title="Monte Carlo Simulation: Portfolio Optimization",
        hover_data=[f"{t} Weight" for t in tickers]
    )
    fig.update_layout(
        xaxis_title="Volatility (Risk)",
        yaxis_title="Expected Return",
        template="plotly_dark"
    )

    return fig, portfolio_data