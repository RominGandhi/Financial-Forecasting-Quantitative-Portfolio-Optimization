import numpy as np
import pandas as pd
import json
from data_fetcher import fetch_data
from portfolio_metrics import calculate_portfolio_metrics, portfolio_return, portfolio_volatility
from scipy.optimize import minimize

def min_volatility(weights, returns):
    return portfolio_volatility(weights, returns)

def max_sharpe_ratio(weights, returns, risk_free_rate=0.01):
    p_metrics = calculate_portfolio_metrics(weights, returns, risk_free_rate)
    return -p_metrics['sharpe_ratio']

def target_return_constraint(weights, returns, target_return):
    portfolio_ret = portfolio_return(weights, returns)
    return portfolio_ret - target_return

def optimize_portfolio(returns, method='sharpe', risk_free_rate=0.01, target_return=None):
    num_assets = len(returns.columns)
    initial_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    if method == 'volatility':
        result = minimize(min_volatility, initial_weights, args=(returns,),
                          method='SLSQP', bounds=bounds, constraints=constraints)
    elif method == 'sharpe':
        result = minimize(max_sharpe_ratio, initial_weights, args=(returns, risk_free_rate),
                          method='SLSQP', bounds=bounds, constraints=constraints)
    elif method == 'target_return' and target_return is not None:
        constraints.append({'type': 'eq', 'fun': lambda x: target_return_constraint(x, returns, target_return)})
        result = minimize(min_volatility, initial_weights, args=(returns,),
                          method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        raise ValueError("Invalid method or missing target for 'target_return'.")

    optimal_weights = result.x
    optimal_metrics = calculate_portfolio_metrics(optimal_weights, returns, risk_free_rate)
    return {'weights': optimal_weights, 'metrics': optimal_metrics}

if __name__ == "__main__":
    # Load configuration from tickers.json
    config_path = 'data/config/tickers.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract tickers, start date, and end date from the config
    tickers = config.get('tickers')
    start_date = config.get('start_date')
    end_date = config.get('end_date')

    # Ensure all necessary data is provided in the JSON configuration
    if not tickers or not start_date or not end_date:
        raise ValueError("tickers, start_date, and end_date must be provided in the tickers.json file.")
    
    # Fetch historical price data using data_fetcher.py
    data = fetch_data(tickers, start_date, end_date)
    
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Optimize for maximum Sharpe ratio
    optimization_result = optimize_portfolio(returns, method='sharpe')
    optimal_weights = optimization_result['weights']
    optimal_metrics = optimization_result['metrics']

    # Print optimal weights and metrics
    print("Optimal Weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight:.2%}")
    
    print(f"Expected Return: {optimal_metrics['return']:.2%}")
    print(f"Volatility: {optimal_metrics['volatility']:.2%}")
    print(f"Sharpe Ratio: {optimal_metrics['sharpe_ratio']:.2f}")
