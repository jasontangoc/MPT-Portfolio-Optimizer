import scipy.optimize as sco

from data_retrieval import *

def calculate_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate=0.0):
    port_return = np.dot(weights, expected_returns)
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_volatility = np.sqrt(port_variance)
    return (port_return - risk_free_rate) / port_volatility

def optimize_portfolio(dataset):
    num_assets = len(dataset.tickers)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    optimum_weights = sco.minimize(lambda w: -calculate_sharpe_ratio(w, dataset.return_vector, dataset.covariance_matrix),
                                   weights,
                                   constraints=constraints,
                                   bounds=bounds
                                   )
    
    return optimum_weights.x
