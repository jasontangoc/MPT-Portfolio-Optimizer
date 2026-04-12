import scipy.optimize as sco

from monte_carlo import *
from data_retrieval import *

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
