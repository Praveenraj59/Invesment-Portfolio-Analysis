import yfinance as yf
import pandas as pd

# Function to download stock data from Yahoo Finance
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Example usage: Get stock data for Apple (AAPL) from 2020 to 2023
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2023-01-01"
stock_data = get_stock_data(ticker, start_date, end_date)

# Save the data to CSV
stock_data.to_csv(f'{ticker}_data.csv')

print(stock_data.head())
