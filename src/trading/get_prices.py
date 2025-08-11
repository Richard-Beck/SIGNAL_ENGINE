
# main.py
import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def download_price_data():
    """
    Connects to the Alpaca API using the new alpaca-py SDK to download 
    minute-level price data for a list of tickers and saves the data to a CSV file.
    """
    # --- 1. Configuration & Setup ---

    # Load environment variables from a .env file in the same directory
    # Your .env file should look like this:
    # APCA_API_KEY_ID=YOUR_KEY_ID
    # APCA_API_SECRET_KEY=YOUR_SECRET_KEY
    load_dotenv()

    # Get API keys from environment variables
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    # Check if keys are loaded
    if not api_key or not secret_key:
        print("Error: Alpaca API keys not found.")
        print("Please create a .env file with your APCA_API_KEY_ID and APCA_API_SECRET_KEY.")
        return

    # Initialize the new Alpaca Data Client
    # The client uses your key type (live or paper) to determine the endpoint.
    data_client = StockHistoricalDataClient(api_key, secret_key)

    # Define file paths
    tickers_file_path = 'data/tickers.txt'
    output_dir = 'data/price_data'

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Load Tickers ---

    try:
        with open(tickers_file_path, 'r') as f:
            # Read tickers and strip any whitespace/newlines
            tickers = [line.strip() for line in f.readlines()]
        print(f"Successfully loaded {len(tickers)} tickers from {tickers_file_path}")
    except FileNotFoundError:
        print(f"Error: Ticker file not found at {tickers_file_path}")
        return

    # --- 3. Fetch Data from Alpaca (New Method) ---

    # Get today's date in YYYY-MM-DD format
    # The SDK handles timezone conversion and market hours automatically.
    today_str = datetime.now().strftime('%Y-%m-%d')
    print(f"\nFetching minute-level price data for {today_str}...")

    try:
        # Create a single request object for all tickers. This is more efficient.
        # *** FIX: Removed the 'end' parameter. ***
        # This tells the API to fetch from the start date until the most recent data.
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Minute,
            start=today_str,
            feed='iex'  # Specify IEX as the data source
        )

        # Make a single API call to get all the data
        bars_data = data_client.get_stock_bars(request_params)

        # The .df property returns a multi-index DataFrame
        final_df = bars_data.df

        if final_df.empty:
            # This message will now appear if the market is closed or it's a holiday.
            print("No data returned for any tickers today (is the market open?).")
            return

        # The 'symbol' is in the index, so we reset it to become a column
        final_df.reset_index(inplace=True)
        
        # Rename 'symbol' to 'ticker' for consistency with the previous script's output
        final_df.rename(columns={'symbol': 'ticker'}, inplace=True)


    except Exception as e:
        print(f"An error occurred while fetching data from Alpaca: {e}")
        return

    # --- 4. Save Data ---

    if final_df.empty:
        print("\nNo data was downloaded. Exiting.")
        return

    # Reorder columns for better readability
    # The timestamp is now a column after reset_index()
    final_df = final_df[['ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Generate a dynamic filename with the current date
    output_filename = f"{today_str}_price_data.csv"
    output_path = os.path.join(output_dir, output_filename)

    # Save the final dataframe to a CSV file, without the pandas index
    final_df.to_csv(output_path, index=False)

    print(f"\nAll data has been successfully saved to: {output_path}")
    print(f"Total rows saved: {len(final_df)}")


if __name__ == '__main__':
    # This allows the script to be run directly from the command line
    download_price_data()
