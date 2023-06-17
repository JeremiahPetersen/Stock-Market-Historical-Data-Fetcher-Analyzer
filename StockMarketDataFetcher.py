import yfinance as yf
import pandas as pd
from datetime import datetime
import logging
from ta import add_all_ta_features
from ta.utils import dropna
import os

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s')

# Define tickers to download
tickers = ['^GSPC', 'SPY', 'DX-Y.NYB', 'TNX', '^VIX', 'GLD']

# Get current date
current_date = datetime.today().strftime('%Y-%m-%d')

# Filenames to save data
filename = 'market_all_historical_data.csv'
filename_raw = 'market_all_historical_data_raw.csv'  # raw data

# Check if files exist
if os.path.isfile(filename):
    # If file exists, load the file and find the latest date
    all_data = pd.read_csv(filename, header=[0, 1], index_col=0, parse_dates=True)
    all_data_raw = pd.read_csv(filename_raw, header=[0, 1], index_col=0, parse_dates=True)
    latest_date = pd.to_datetime(all_data.index).max()

    # Define the start date for new data as the day after the latest date in the file
    start_date = (latest_date + pd.DateOffset(1)).strftime('%Y-%m-%d')
else:
    # If file doesn't exist, fetch all historical data
    all_data = pd.DataFrame()
    all_data_raw = pd.DataFrame()  # raw data
    start_date = "1900-01-01"

failed_tickers = []  # Keep track of tickers which failed during technical analysis
updated = False  # Flag to check if data was updated
for ticker in tickers:
    logging.info(f'Starting download for {ticker}')
    try:
        # Download historical data as DataFrame
        data = yf.download(ticker, start=start_date, end=current_date)

        if data.empty:
            logging.info(f'No new data for {ticker}')
            continue

        # Create MultiIndex columns for raw data
        raw_data = data.copy()
        raw_data.columns = pd.MultiIndex.from_product([[ticker], raw_data.columns])

        # Merge raw data to the raw DataFrame
        if all_data_raw.empty:
            all_data_raw = raw_data
        else:
            all_data_raw = all_data_raw.join(raw_data, how='outer')

        # Ensure the data is clean for technical analysis
        data = dropna(data)

        # Add all technical analysis features
        data = add_all_ta_features(
            data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

        # Create MultiIndex columns
        data.columns = pd.MultiIndex.from_product([[ticker], data.columns])

        # Merge the dataframes on Date index
        if all_data.empty:
            all_data = data
        else:
            all_data = all_data.join(data, how='outer')

        logging.info(f'Completed download for {ticker}')
        updated = True  # Set flag to True as data was updated
    except Exception as e:
        logging.error(f'Error occurred while downloading data for {ticker}: {str(e)}')
        failed_tickers.append(ticker)

# Remove tickers which failed during technical analysis
for ticker in failed_tickers:
    if ticker in all_data.columns.get_level_values(0):
        all_data.drop(ticker, axis=1, level=0, inplace=True)

# Save to CSV only if data was updated
if updated:
    all_data.to_csv(filename)
    all_data_raw.to_csv(filename_raw)  # raw data
    logging.info('All data and raw data saved to CSV files')
else:
    logging.info('No new data was downloaded today.')
