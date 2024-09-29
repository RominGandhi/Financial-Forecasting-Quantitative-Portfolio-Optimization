
import yfinance as yf
import pandas as pd
import os
import json
import requests

# Define the directory where data will be saved
DATA_DIR = os.path.join(os.getcwd(), 'data', 'historical_data')
CONFIG_DIR = os.path.join(os.getcwd(), 'data', 'config')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# API keys for different data sources
ALPHA_VANTAGE_API_KEY =   # Alpha Vantage API key
CRYPTOCOMPARE_API_KEY =   #  Cryptocompare API key


def get_asset_name(ticker):
    """Get the full name of the asset from its ticker."""
    try:
        asset = yf.Ticker(ticker)
        return asset.info['longName']
    except KeyError:
        return ticker  # Return the ticker if the name is not found


def fetch_data_yfinance(tickers, start_date, end_date):
    """Fetch data from Yahoo Finance for stocks and ETFs."""
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def fetch_data_alpha_vantage(ticker, start_date=None, end_date=None):
    """Fetch data from Alpha Vantage for bonds."""
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" in data:
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={"4. close": ticker})
        df = df[[ticker]]
        df = df.sort_index()
        if start_date and end_date:
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        return df
    else:
        print(f"Error fetching data for {ticker} from Alpha Vantage.")
        return pd.DataFrame()
    
def get_alpha_vantage_asset_name(ticker):
    """Get the full name of the asset from Alpha Vantage given a ticker."""
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()

    return data.get("Name", ticker)  # Return the name or the ticker if not found

def get_cryptocompare_asset_name(crypto):
    """Get the full name of the cryptocurrency from CryptoCompare given its symbol."""
    url = f"https://min-api.cryptocompare.com/data/all/coinlist?api_key={CRYPTOCOMPARE_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if crypto in data["Data"]:
        return data["Data"][crypto].get("CoinName", crypto)
    return crypto  # Return the crypto symbol if the name is not found



def fetch_data_cryptocompare(crypto, start_date=None, end_date=None):
    """Fetch data from Cryptocompare for cryptocurrencies."""
    url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={crypto}&tsym=USD&limit=2000&api_key={CRYPTOCOMPARE_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if data['Response'] == 'Success':
        df = pd.DataFrame(data['Data']['Data'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df.rename(columns={'close': crypto})
        df = df[[crypto]]
        if start_date and end_date:
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        return df
    else:
        print(f"Error fetching data for {crypto} from Cryptocompare.")
        return pd.DataFrame()

def fetch_data(tickers, start_date, end_date):
    """Fetch data for various financial instruments based on their category."""
    stock_etf_tickers = []
    bond_tickers = []
    crypto_tickers = []

    # Categorize tickers based on suffixes/predefined rules
    for ticker in tickers:
        ticker = ticker.strip().upper()
        if ticker.endswith('.BOND'):
            bond_tickers.append(ticker.replace('.BOND', ''))
        elif ticker.endswith('.CRYPTO'):
            crypto_tickers.append(ticker.replace('.CRYPTO', ''))
        else:
            stock_etf_tickers.append(ticker)

    data = pd.DataFrame()

    # Fetch data for stocks and ETFs
    if stock_etf_tickers:
        stock_etf_data = fetch_data_yfinance(stock_etf_tickers, start_date, end_date)
        if not stock_etf_data.empty:
            data = data.join(stock_etf_data, how='outer') if not data.empty else stock_etf_data

    # Fetch data for bonds
    if bond_tickers:
        for bond in bond_tickers:
            bond_data = fetch_data_alpha_vantage(bond, start_date, end_date)
            if not bond_data.empty:
                data = data.join(bond_data, how='outer')

    # Fetch data for cryptocurrencies
    if crypto_tickers:
        for crypto in crypto_tickers:
            crypto_data = fetch_data_cryptocompare(crypto, start_date, end_date)
            if not crypto_data.empty:
                data = data.join(crypto_data, how='outer')


    data = data.ffill().bfill()
    return data


def save_data(data, filename):
    """Save the fetched data to a CSV file."""
    filepath = os.path.join(DATA_DIR, filename)
    data.to_csv(filepath)

def load_data(filename):
    """Load data from a CSV file."""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        print(f"File {filename} does not exist.")
        return None

def save_config(tickers, start_date, end_date):
    """Save the configuration of tickers and date range to a JSON file."""
    config = {
        'tickers': tickers,
        'start_date': start_date,
        'end_date': end_date
    }
    config_path = os.path.join(CONFIG_DIR, 'tickers.json')
    with open(config_path, 'w') as f:
        json.dump(config, f)

def load_config():
    """Load the configuration of tickers and date range from a JSON file."""
    config_path = os.path.join(CONFIG_DIR, 'tickers.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        print("Config file does not exist.")
        return None

if __name__ == "__main__":
    # Ask user for input
    tickers_input = input("Enter the tickers separated by commas (e.g., AAPL,IEF.BOND,BTC.CRYPTO) - DO NOT ENTER DIFFERENT FINANCIAL INSTRUMENTS TOGETHER : ")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    
    
    # Fetch and display the full names of the assets
    print("\nAsset Names:")
    for ticker in tickers:
        full_name = get_asset_name(ticker)
        print(f"{ticker}: {full_name}")

    
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    
    # Save the configuration
    save_config(tickers, start_date, end_date)
    
    # Fetch data
    data = fetch_data(tickers, start_date, end_date)
    
    # Save data
    save_data(data, 'portfolio_data.csv')
    
    # Load data
    loaded_data = load_data('portfolio_data.csv')
    print("Loaded Data:\n", loaded_data.head())
    
