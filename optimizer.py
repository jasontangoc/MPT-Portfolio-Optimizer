import scipy.optimize as sco

from monte_carlo import *

def optimize_portfolio(dataset):
    tickers = dataset.columns
    num_assets = len(tickers)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))

    optimum_weights = sco.minimize(-calculate_sharpe_ratio, weights, 
                                   args=(dataset.return_vector, dataset.covariance_matrix),
                                   constraints=constraints,
                                   bounds=bounds
                                   )
    
    return optimum_weights

print(optimize_portfolio(df))
