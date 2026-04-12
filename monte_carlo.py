import numpy as np
import pandas as pd
import plotly.express as ex

# Generate random portfolios and plot the efficient frontier
def run_monte_carlo(tickers, expected_returns, cov_matrix, num_portfolios=10000):

    num_assets = len(tickers) 
    
    results_returns = []
    results_volatility = []
    results_weights = []

    # Monte Carlo Loop
    for _ in range(num_portfolios):
        # Generate random weights and normalize weights so they sum up to 1
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights) 
        
        # Expected return 
        port_return = np.dot(weights, expected_returns)
        
        # Volatility 
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_volatility = np.sqrt(port_variance)
        
        # Store the results
        results_returns.append(port_return)
        results_volatility.append(port_volatility)
        results_weights.append(weights)

    # Convert to a Pandas DataFrame for easy plotting
    portfolio_data = pd.DataFrame({
        "Return": results_returns,
        "Volatility": results_volatility
    })

    # Add the individual stock weights as columns for the hover data
    for counter, ticker in enumerate(tickers):
        portfolio_data[f"{ticker} Weight"] = [w[counter] for w in results_weights]

    # Calculate sharpe ratio assuming a 0% Risk-Free Rate
    portfolio_data["Sharpe Ratio"] = portfolio_data["Return"] / portfolio_data["Volatility"]

    # Plot
    fig = ex.scatter(
        portfolio_data, 
        x="Volatility", 
        y="Return", 
        color="Sharpe Ratio",
        color_continuous_scale="Viridis",
        title="Monte Carlo Simulation: Portfolio Optimization",
        hover_data=[f"{t} Weight" for t in tickers] # Shows weights when you hover over a dot
    )
    
    # Clean up 
    fig.update_layout(
        xaxis_title="Volatility (Risk)",
        yaxis_title="Expected Return",
        template="plotly_dark" 
    )
    
    return fig, portfolio_data