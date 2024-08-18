import numpy as np
import pandas as pd
import json
from data_fetcher import fetch_data

def calculate_annualized_return(returns, periods_per_year=252):
    compounded_growth = (1 + returns).prod()
    n_periods = returns.shape[0]
    annualized_return = compounded_growth**(periods_per_year / n_periods) - 1
    return annualized_return

def calculate_annualized_volatility(returns, periods_per_year=252):
    annualized_volatility = returns.std() * np.sqrt(periods_per_year)
    return annualized_volatility

def calculate_sharpe_ratio(returns, risk_free_rate=0.01, periods_per_year=252):
    excess_return = returns.mean() * periods_per_year - risk_free_rate
    annualized_volatility = calculate_annualized_volatility(returns, periods_per_year)
    sharpe_ratio = excess_return / annualized_volatility
    return sharpe_ratio

def portfolio_return(weights, returns):
    return np.sum(weights * returns.mean()) * returns.shape[0]

def portfolio_volatility(weights, returns):
    covariance_matrix = returns.cov()
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance) * np.sqrt(returns.shape[0])
    return portfolio_volatility

def calculate_portfolio_metrics(weights, returns, risk_free_rate=0.01):
    p_return = portfolio_return(weights, returns)
    p_volatility = portfolio_volatility(weights, returns)
    p_sharpe = (p_return - risk_free_rate) / p_volatility
    return {'return': p_return, 'volatility': p_volatility, 'sharpe_ratio': p_sharpe}

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
    
    # Ask the user for weights
    weights = []
    for ticker in tickers:
        while True:
            try:
                weight = float(input(f"Enter the weight for {ticker} (as a fraction, e.g., 0.2 for 20%): "))
                weights.append(weight)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    # Ensure the weights sum to 1
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("The sum of the weights must be 1.")

    # Fetch historical price data using data_fetcher.py
    data = fetch_data(tickers, start_date, end_date)
    
    # Calculate daily returns
    returns = data.pct_change().dropna()
    
    # Calculate portfolio metrics
    weights = np.array(weights)
    metrics = calculate_portfolio_metrics(weights, returns)
    
    # Convert to percentage format
    expected_return_percentage = metrics['return'] * 100
    volatility_percentage = metrics['volatility'] * 100
    
    # Output the results with explanations
    print(f"Portfolio Expected Return: {expected_return_percentage:.2f}%")
    print("This represents the annualized return of the portfolio, based on historical data and the provided weights.")
    
    print(f"Portfolio Volatility: {volatility_percentage:.2f}%")
    print("This represents the annualized risk or standard deviation of the portfolio's returns. It indicates the variability of returns.")
    
    print(f"Portfolio Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print("This measures the risk-adjusted return of the portfolio. A higher Sharpe ratio indicates better risk-adjusted performance.")
