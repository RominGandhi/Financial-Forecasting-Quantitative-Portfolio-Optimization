
# 📈 Financial Forecasting & Quantitative Portfolio Optimization

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14.0-orange.svg)](https://www.tensorflow.org/)  
[![Alpha Vantage](https://img.shields.io/badge/Alpha--Vantage-API-green.svg)](https://www.alphavantage.co/)
[![CryptoCompare](https://img.shields.io/badge/CryptoCompare-API-green.svg)](https://www.cryptocompare.com/)

This project not only predicts future stock prices but also provides **portfolio optimization** using various risk metrics such as the Sharpe ratio and volatility. It integrates financial data from APIs, processes time-series data, and generates optimal portfolio allocations using advanced machine learning techniques.

> Note: This project is currently in progress. Several bugs and limitations may exist, and I are actively working on improving performance and accuracy. Future enhancements will include better support for additional financial assets such as Bitcoin, bonds, and more sophisticated prediction models.

## 🎯 Project Overview

This project is designed to assist users in analyzing financial instruments, predicting their future prices using an LSTM model, and optimizing portfolio allocation to maximize returns and minimize risk. 

Key features include:
- **Stock Price Prediction** using Long Short-Term Memory (LSTM) models.
- **Portfolio Optimization** using mathematical techniques (e.g., maximizing the Sharpe ratio).
- **Exchange holiday handling** via `pandas_market_calendars` to exclude non-trading days.
- **Dynamic visualization** for actual vs predicted prices, and portfolio metrics.

---

## 🔧 Tech Stack

- **Language**: Python 3.8+
- **Libraries**: TensorFlow, Pandas, NumPy, Matplotlib, Alpha Vantage API, Yahoo Finance API, SciPy
- **Tools**: LSTM Neural Networks, Portfolio Optimization via SLSQP

---

## 🚀 Project Structure

```
📦 Financial-Instruments-Predictive-Analytics-Portfolio-Optimization
 ┣ 📂 data
 ┃ ┣ 📂 config                    # Configuration data (if applicable)
 ┃ ┣ 📂 historical_data           # Historical stock data CSV files
 ┃ ┗ 📂 Predicted Prices          # Folder to save predicted prices CSVs
 ┣ 📂 src                         # Source code files
 ┃ ┣ 📂 __pycache__               # Compiled Python files
 ┃ ┣ 📜 data_fetcher.py           # Fetch financial data from APIs
 ┃ ┣ 📜 exchange_mapping.py       # Map exchange codes for holiday handling
 ┃ ┣ 📜 generate_doc.py           # Generate reports for portfolio analysis
 ┃ ┣ 📜 main.py                   # Main execution script
 ┃ ┣ 📜 optimization.py           # Portfolio optimization methods
 ┃ ┣ 📜 portfolio_metrics.py      # Portfolio metric calculations
 ┃ ┗ 📜 predict_prices.py         # Predict stock prices using LSTM
 ┣ 📂 user_log                    # Log directory for user activities
 ┗ 📜 README.md                   # This file
 ┗ 📜 requirements.txt            # All the Python libraries and specific versions that are essential for running the project
 ┗ 📜 LICENSE                     # You are free to use, copy, modify, merge, publish, distribute, and sublicense the project, as long as the original copyright and permission notice are included
```

---

## 📊 Features Breakdown

### 1. Stock Price Prediction
The LSTM model processes historical stock data and predicts future prices. The prediction accounts for weekends and holidays using real-time data fetched from Yahoo Finance and Alpha Vantage APIs.

- **LSTM Architecture**: The model uses two layers of LSTM with Dropout regularization to prevent overfitting.
- **Prediction Horizons**: You can predict stock prices for a configurable number of days.
- **Handling Non-Trading Days**: It intelligently skips weekends and holidays based on the asset’s exchange.

### 2. Portfolio Optimization
The project offers portfolio optimization using the **Sharpe Ratio** and **Minimum Volatility** methods. The optimization algorithms are implemented using the SLSQP method from SciPy’s optimization package.

- **Sharpe Ratio Maximization**: Optimizes the portfolio to achieve the highest return-to-risk ratio.
- **Volatility Minimization**: Aims to create a portfolio with the lowest possible risk for a given set of assets.

---

## 🛠 Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RominGandhi/Financial-Instruments-Predictive-Analytics-Portfolio-Optimization.git
   cd Financial-Instruments-Predictive-Analytics-Portfolio-Optimization
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API keys**:
   - Replace the placeholder `ALPHA_VANTAGE_API_KEY` in `data_fetcher.py` with your actual API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key).
   - Replace the placeholder `CRYPTOCOMPARE_API_KEY` in in `data_fetcher.py` with your actual API key from [CryptoCompare](https://min-api.cryptocompare.com/).

4. **Run the main script**:
   ```bash
   python main.py
   ```

---

## 📈 How It Works

### Data Fetching & Preprocessing
The project fetches historical stock prices and exchange data using Yahoo Finance, Alpha Vantage and CryptoCompare. The data is then preprocessed, with weekends and holidays removed based on the exchange calendar for each asset ticker.

### LSTM Model for Stock Price Prediction
The LSTM model is trained on historical data to predict future stock prices. You can select how many days ahead to predict, and the model will output both actual and predicted stock prices within an graph.

### Portfolio Optimization
The optimization module (`optimization.py`) uses **numerical optimization** to determine the best asset weights for either minimizing risk or maximizing the Sharpe Ratio.

### Generating Reports
The project includes a module to generate detailed reports and csv file with portfolio analysis, including predicted asset prices and optimal portfolio allocations. These files are saved as `.docx` or `.csv`files for easy sharing.

---

## 📝 Future Enhancements

- [ ] Add more sophisticated models (e.g., GRU, Transformer) for stock price prediction.
- [ ] Implement real-time portfolio rebalancing strategies.
- [ ] Incorporate alternative financial data (e.g., sentiment analysis from news articles).
- [ ] Add a web-based interface for easier access to the tool.

---

## 🧠 Insights & Challenges

- **Handling Market Closures**: By using `pandas_market_calendars`, the project accurately skips weekends and market holidays, providing more reliable predictions.
- **Optimization Techniques**: Portfolio optimization is crucial for balancing returns and risk. Fine-tuning the optimization algorithm led to better performance.

---

## 👨‍💻 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check out the [issues page](https://github.com/RominGandhi/Financial-Instruments-Predictive-Analytics-Portfolio-Optimization/issues).

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🛡️ Disclaimer

This project is for educational purposes only. Stock price prediction and portfolio optimization are complex tasks, and the results produced by this tool should **not** be used for actual trading or investment decisions.

---

## 🙌 Acknowledgments

- **Alpha Vantage** for providing the free API for fetching financial data.
- **Yahoo Finance** for real-time stock market data.
- **CryptoCompare** for providing cryptocurrency data through their API.
- Open-source developers for the amazing libraries that make this project possible!
