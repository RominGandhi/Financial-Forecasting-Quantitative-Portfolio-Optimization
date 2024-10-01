import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import importlib
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import pandas_market_calendars as mcal

# Importing exchange mapping from the new file
from exchange_mapping import exchange_mapping

# Dynamically importing necessary classes using importlib
Sequential = importlib.import_module('tensorflow.keras.models').Sequential
LSTM = importlib.import_module('tensorflow.keras.layers').LSTM
Dense = importlib.import_module('tensorflow.keras.layers').Dense
Dropout = importlib.import_module('tensorflow.keras.layers').Dropout

# Alpha Vantage API Key
ALPHA_VANTAGE_API_KEY = 'OJ8TCL50HAAKMZDN'  
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY)

# Setting the default path for the historical data directory
DATA_DIR = os.path.join(os.getcwd(), 'data', 'historical_data')

# Path to the directory where predicted prices CSV files will be saved
PREDICTED_DIR = os.path.join(os.getcwd(), 'data', 'Predicted Prices')

def get_exchange_for_ticker(ticker):
    """Fetch exchange information for a given ticker using Yahoo Finance or CryptoCompare for cryptocurrencies."""
    try:
        if ticker.endswith('.CRYPTO'):
            # Handle cryptocurrencies separately
            exchange_full_name = "Crypto Market"
            print(f"Exchange for {ticker}: {exchange_full_name}")
            return exchange_full_name
        else:
            # Fetch exchange info for stocks/ETFs from Yahoo Finance
            stock_info = yf.Ticker(ticker).info
            exchange_abbr = stock_info.get('exchange')
            exchange_full_name = exchange_mapping.get(exchange_abbr, exchange_abbr)  # Map abbreviation to full name
            if exchange_full_name:
                print(f"Exchange for {ticker}: {exchange_full_name}")
            else:
                print(f"Exchange for {ticker} not found.")
            return exchange_abbr
    except Exception as e:
        print(f"Error fetching exchange for ticker {ticker}: {e}")
        return None

def get_exchange_calendar(ticker):
    """Determine the trading calendar based on the asset's ticker."""
    exchange = get_exchange_for_ticker(ticker)
    
    # Mapping between exchange abbreviations and pandas_market_calendars exchange codes
    exchange_map = {
        'NMS': 'XNYS',   # Nasdaq or NYSE
        'NYSE': 'XNYS',  # New York Stock Exchange
        'BOND': 'XBOND', # US Bond Market
        'TSE': 'XTSE',   # Toronto Stock Exchange
        'TOR': 'XTSE',   # Toronto Stock Exchange (Alternate symbol)
        'CVE': 'XTSE',   # TSX Venture Exchange
        'LSE': 'XLON',   # London Stock Exchange
        'FRA': 'XFRA',   # Frankfurt Stock Exchange
        'ETR': 'XETR',   # Xetra (Germany)
        'SWX': 'XSWX',   # Swiss Exchange
        'HKG': 'XHKG',   # Hong Kong Stock Exchange
        'TYO': 'XTKS',   # Tokyo Stock Exchange
        'ASX': 'XASX',   # Australian Securities Exchange
        'SGX': 'XSES',   # Singapore Exchange
        'KRX': 'XKRX',   # Korea Exchange
        'BSE': 'XBOM',   # Bombay Stock Exchange
        'NSE': 'XNSE',   # National Stock Exchange of India
        'SHE': 'XSHE',   # Shenzhen Stock Exchange
        'SHG': 'XSHG',   # Shanghai Stock Exchange
    }

    exchange_code = exchange_map.get(exchange)
    
    if exchange_code:
        return mcal.get_calendar(exchange_code)
    else:
        print(f"Exchange for ticker {ticker} not found in mapping. Assuming no holidays.")
        return None

def get_trading_days(exchange_calendar, start_date, end_date):
    """Get valid trading days between two dates, excluding weekends and holidays."""
    if exchange_calendar:
        # Get valid trading days considering exchange holidays
        schedule = exchange_calendar.valid_days(start_date=start_date, end_date=end_date)
        return schedule
    else:
        # If no exchange calendar is found, exclude weekends (Saturday, Sunday)
        date_range = pd.date_range(start=start_date, end=end_date)
        trading_days = date_range[date_range.weekday < 5]  # Exclude weekends
        return trading_days

def load_historical_data(filename):
    """Load historical data from a CSV file, only considering tickers and ignoring exchanges."""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Loading file: {filepath}")
        
        # Read the first row to extract tickers (ignoring exchange names)
        tickers = pd.read_csv(filepath, nrows=1, index_col=0).columns
        print(f"Tickers found: {list(tickers)}")

        # Load the actual data, skipping the first two rows
        data = pd.read_csv(filepath, skiprows=2, index_col=0, parse_dates=True)
        
        # Assign the tickers as column names
        data.columns = tickers
        
        # Clean the data by converting non-numeric values to NaN and dropping rows with NaN
        data = data.apply(pd.to_numeric, errors='coerce').dropna()

        print("DataFrame loaded and cleaned:")
        print(data.head())  # Show the cleaned data
        return data, tickers
    else:
        print(f"File {filename} does not exist.")
        return None, None

def preprocess_data(data, seq_length=60):
    """Normalize the data and create sequences."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i])
        y.append(scaled_data[i, 0])  # Predicting the next closing price

    X, y = np.array(X), np.array(y)
    return X, y, scaler

def build_lstm_model(input_shape):
    """Build a LSTM model."""
    model = Sequential()
    
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=25))
    model.add(Dense(units=1))  # Output layer predicting the next price
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(data, model, epochs=20, batch_size=32):
    """Train the LSTM model."""
    X, y, scaler = preprocess_data(data)
    
    # Split into training and testing sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to numpy arrays if not already
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    
    # Predicting on the test set
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate the root mean squared error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"RMSE: {rmse}")
    
    return model, scaler, predictions, y_test

def predict_future_prices_lstm(model, data, scaler, ticker, future_days=30):
    """Predict future prices using the trained LSTM model, excluding weekends and holidays."""
    seq_length = 60
    last_sequence = data[-seq_length:].values
    last_sequence_scaled = scaler.transform(last_sequence)
    
    predicted_prices = []
    prediction_dates = []
    
    current_date = data.index[-1] + pd.Timedelta(days=1)

    # Get exchange calendar based on ticker
    exchange_calendar = get_exchange_calendar(ticker)

    # Determine the range of future dates
    end_date = current_date + pd.Timedelta(days=future_days * 2)  # Buffer for weekends/holidays
    trading_days = get_trading_days(exchange_calendar, current_date, end_date)
    
    for date in trading_days[:future_days]:
        X_pred = last_sequence_scaled.reshape((1, seq_length, -1))
        predicted_price_scaled = model.predict(X_pred)

        predicted_price = scaler.inverse_transform(predicted_price_scaled)[0, 0]
        predicted_prices.append(predicted_price)
        prediction_dates.append(date)

        # Update the last sequence with the new predicted price
        last_sequence_scaled = np.append(last_sequence_scaled[1:], predicted_price_scaled).reshape(-1, 1)

    return pd.DataFrame(predicted_prices, index=prediction_dates, columns=['Predicted Price'])

if __name__ == "__main__":
    
    # Check if DATA_DIR exists
    if not os.path.exists(DATA_DIR):
        print(f"The directory {DATA_DIR} does not exist.")
    else:
        # List available files in the directory, excluding hidden files
        files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f)) and not f.startswith('.')]
        
        if not files:
            print("No CSV files found in the directory.")
        else:
            # Display available files and prompt the user to select one
            print("Available files:")
            for idx, file in enumerate(files):
                print(f"{idx}: {file}")
            
            # Input validation for selecting a file
            while True:
                try:
                    file_idx = int(input("Select the file number you want to use: "))
                    if 0 <= file_idx < len(files):
                        break
                    else:
                        print(f"Please enter a number between 0 and {len(files) - 1}.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
            
            selected_file = files[file_idx]
            
            historical_data, tickers = load_historical_data(selected_file)
            
            if historical_data is not None:
                # Fetch and print exchange information for each ticker
                for ticker in tickers:
                    print(f"\nFetching exchange information for ticker: {ticker}")
                    get_exchange_for_ticker(ticker)
                
                # Ask the user how many days to predict
                future_days = int(input("\nEnter the number of days to predict: "))
                
                # Loop through each column (asset) in the data
                for column in historical_data.columns:
                    print(f"\nProcessing asset: {column}")
                    
                    model = build_lstm_model((60, 1))
                    trained_model, scaler, predictions, y_test = train_lstm_model(historical_data[[column]], model)
                    
                    # Plotting the actual vs predicted prices
                    plt.figure(figsize=(12,6))
                    plt.plot(y_test, color='blue', label='Actual Stock Price')
                    plt.plot(predictions, color='red', label='Predicted Stock Price')
                    plt.title(f'Stock Price Prediction for {column}')
                    plt.xlabel('Time')
                    plt.ylabel('Stock Price')
                    plt.legend()
                    plt.show()
                    
                    future_prices = predict_future_prices_lstm(trained_model, historical_data[[column]], scaler, column, future_days=future_days)
                    print(f"Future prices for {column}:")
                    print(future_prices)
                    
                    # Ensure the output directory exists
                    os.makedirs(PREDICTED_DIR, exist_ok=True)
                    
                    # Save the future prices to a CSV file in the specified directory
                    output_file = os.path.join(PREDICTED_DIR, f"{column}_future_prices.csv")
                    future_prices.to_csv(output_file)
                    print(f"Predicted prices saved to {output_file}")
