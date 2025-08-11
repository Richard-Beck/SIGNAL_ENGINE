# main_analyzer.py
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, time
from pandas import Timedelta
import pytz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm # Import tqdm for the progress bar

def plot_top_narratives_performance(results_df, prices_df, final_summary, today_str):
    """
    Identifies the top 3 performing narratives and plots the normalized price action
    of their associated tickers, highlighting all predictions and their outcomes.
    """
    print("\nGenerating performance plots for the top 3 narratives...")

    try:
        # 1. Identify the top 3 narratives for the EOD window by Confidence score
        if 'EOD' not in final_summary.columns.get_level_values(0):
            print("EOD results not available for plotting.")
            return
            
        top_narratives = final_summary['EOD'].sort_values(by='Confidence', ascending=False).head(3)
        
        # Loop through each of the top 3 narratives
        for top_narrative_id, narrative_stats in top_narratives.iterrows():
            print(f"\nPlotting for: '{top_narrative_id}' (EOD Hit Rate: {narrative_stats['Hit Rate']:.2f}%)")

            # 2. Filter data for the current narrative
            top_narrative_results = results_df[results_df['narrative_id'] == top_narrative_id].copy()
            tickers_to_plot = top_narrative_results['ticker'].unique()
            prices_to_plot = prices_df[prices_df.index.get_level_values('ticker').isin(tickers_to_plot)].copy()

            # 3. Normalize prices
            prices_to_plot['normalized'] = prices_to_plot['close'] / prices_to_plot.groupby(level='ticker')['close'].transform('first')

            # 4. Create the plot
            fig, ax = plt.subplots(figsize=(18, 10))

            # Plot the normalized price for each ticker
            for ticker, group in prices_to_plot.groupby(level='ticker'):
                ax.plot(group.index.get_level_values('timestamp'), group['normalized'], label=ticker, alpha=0.5)

            # 5. Add markers for ALL predictions (hits and misses)
            for _, row in top_narrative_results.iterrows():
                signal_time = row['signal_time_utc']
                ticker = row['ticker']
                prediction = row['prediction']
                is_hit = row['hit_EOD']
                
                marker_shape = '^' if prediction == 'LONG' else 'v'
                marker_color = 'green' if is_hit else 'red'
                
                try:
                    price_at_signal_pos = prices_to_plot.loc[ticker].index.get_indexer([signal_time], method='nearest')[0]
                    normalized_price_at_signal = prices_to_plot.loc[ticker].iloc[price_at_signal_pos]['normalized']
                    ax.plot(signal_time, normalized_price_at_signal, marker=marker_shape, color=marker_color, markersize=10, markeredgecolor='black', linestyle='None')
                except (KeyError, IndexError):
                    continue
            
            # 6. Format the plot
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='^', color='green', label='Long Hit', markersize=10, linestyle='None', markeredgecolor='k'),
                Line2D([0], [0], marker='v', color='green', label='Short Hit', markersize=10, linestyle='None', markeredgecolor='k'),
                Line2D([0], [0], marker='^', color='red', label='Long Miss', markersize=10, linestyle='None', markeredgecolor='k'),
                Line2D([0], [0], marker='v', color='red', label='Short Miss', markersize=10, linestyle='None', markeredgecolor='k')
            ]
            main_legend = ax.legend(loc='upper left', title='Tickers')
            ax.add_artist(main_legend)
            ax.legend(handles=legend_elements, loc='upper right', title='Predictions')

            ax.set_title(f"Performance of Narrative: '{top_narrative_id}' on {today_str}", fontsize=20)
            ax.set_ylabel("Price (Normalized to 1 at Open)", fontsize=14)
            ax.set_xlabel("Time (UTC)", fontsize=14)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=pytz.utc))
            plt.xticks(rotation=45)
            fig.tight_layout()

            # 7. Save the plot with a unique name
            safe_narrative_id = "".join(c for c in top_narrative_id if c.isalnum()).rstrip() # Sanitize filename
            plot_filename = f'reports/{today_str}_{safe_narrative_id}_performance.png'
            plt.savefig(plot_filename)
            plt.close(fig)
            print(f"Plot saved successfully to {plot_filename}")

    except (IndexError, KeyError) as e:
        print(f"Could not generate plot. No valid top narrative found or data was missing. Error: {e}")


def analyze_narrative_performance():
    """
    Loads narrative signals and price data to analyze the predictive
    performance of the signals over different forward-looking time windows.
    """
    # --- 1. Configuration & Setup ---
    MIN_PREDICTIONS = 20
    match_file_path = 'data/match_file.txt'
    today_str = datetime.now().strftime('%Y-%m-%d')
    price_file_path = f'data/price_data/{today_str}_price_data.csv'

    if not os.path.exists(price_file_path):
        print(f"Error: Price data file not found at {price_file_path}")
        return

    utc_tz = pytz.utc
    edt_tz = pytz.timezone('America/New_York')
    market_open = time(9, 30)
    market_close = time(16, 0)

    print("Starting narrative performance analysis...")
    
    # --- 2. Load and Prepare Data ---
    print(f"Loading price data from {price_file_path}...")
    prices_df = pd.read_csv(price_file_path)
    prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp']).dt.tz_convert(utc_tz)
    prices_df.set_index(['ticker', 'timestamp'], inplace=True)
    prices_df.sort_index(inplace=True)
    print("Price data loaded and indexed.")

    try:
        with open(match_file_path, 'r') as f:
            narratives = [json.loads(line) for line in f]
        print(f"Loaded {len(narratives)} narrative events from {match_file_path}.")
    except FileNotFoundError:
        print(f"Error: Match file not found at {match_file_path}")
        return

    # --- 3. Process Each Signal and Test Performance ---
    results = []
    market_close_utc = edt_tz.localize(datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)).astimezone(utc_tz)

    # *** NEW: Wrap the main loop with tqdm for a progress bar ***
    for narrative in tqdm(narratives, desc="Processing Signals", unit="narrative"):
        confirming = narrative.get('confirming_matches', 0)
        refuting = narrative.get('refuting_matches', 0)
        
        signal_time_edt_dt = datetime.fromisoformat(narrative['timestamp'])
        
        if not (market_open <= signal_time_edt_dt.time() <= market_close):
            continue

        signal_time_utc = edt_tz.localize(signal_time_edt_dt).astimezone(utc_tz)

        for item in narrative['affected_tickers']:
            ticker = item['ticker']
            position_if_true = item['position_if_true']
            final_prediction = None

            if confirming > 0 and refuting == 0:
                final_prediction = position_if_true
            elif refuting > 0 and confirming == 0:
                final_prediction = 'SHORT' if position_if_true == 'LONG' else 'LONG'
            else:
                continue

            try:
                prices_for_ticker = prices_df.loc[ticker]
                signal_price_pos = prices_for_ticker.index.get_indexer([signal_time_utc], method='nearest')[0]
                start_price = prices_for_ticker.iloc[signal_price_pos]['close']

                windows = {'T+15m': signal_time_utc + Timedelta(minutes=15), 'T+60m': signal_time_utc + Timedelta(minutes=60), 'EOD': market_close_utc}
                
                result_row = {'narrative_id': narrative['narrative_id'], 'ticker': ticker, 'prediction': final_prediction, 'signal_time_utc': signal_time_utc}

                for window_name, end_time in windows.items():
                    end_price_pos = prices_for_ticker.index.get_indexer([end_time], method='nearest')[0]
                    end_price = prices_for_ticker.iloc[end_price_pos]['close']
                    price_change_pct = ((end_price - start_price) / start_price) * 100
                    is_hit = ((final_prediction == 'LONG' and price_change_pct > 0) or (final_prediction == 'SHORT' and price_change_pct < 0))
                    result_row[f'hit_{window_name}'] = is_hit
                results.append(result_row)
            except (KeyError, IndexError):
                continue

    if not results:
        print("No valid, testable signals were found in the match file.")
        return

    # --- 4. Aggregate and Summarize Results ---
    print("\nAggregating results...")
    results_df = pd.DataFrame(results)
    summary_dfs = []
    for window in ['T+15m', 'T+60m', 'EOD']:
        hit_col = f'hit_{window}'
        summary = results_df.groupby('narrative_id').agg(Predictions=(hit_col, 'count'), Hits=(hit_col, 'sum')).astype(int)
        summary = summary[summary['Predictions'] >= MIN_PREDICTIONS]
        if summary.empty: continue
        summary['Misses'] = summary['Predictions'] - summary['Hits']
        summary['Hit Rate'] = (summary['Hits'] / summary['Predictions'] * 100)
        summary['Confidence'] = (summary['Hit Rate'] - 50) * np.log10(summary['Predictions'])
        summary.columns = pd.MultiIndex.from_product([[window], summary.columns])
        summary_dfs.append(summary)

    if not summary_dfs:
        print(f"No narratives met the minimum prediction threshold of {MIN_PREDICTIONS}.")
        return

    final_summary = pd.concat(summary_dfs, axis=1)
    
    # --- 5. Display Enhanced Report ---
    print("\n--- Narrative Performance Report ---")
    print(f"(Showing narratives with at least {MIN_PREDICTIONS} predictions)\n")
    for window in ['T+15m', 'T+60m', 'EOD']:
        if window not in final_summary.columns.get_level_values(0): continue
        sorted_summary = final_summary[window].sort_values(by='Confidence', ascending=False)
        print(f"--- Top 10 Performers by Confidence Score ({window}) ---")
        print(sorted_summary.head(10).round(2).to_string())
        print("\n" + "="*50 + "\n")

    report_filename = f'reports/{today_str}_narrative_report.csv'
    os.makedirs('reports', exist_ok=True)
    final_summary.round(2).to_csv(report_filename)
    print(f"Full report saved to {report_filename}")

    # --- 6. Generate Plot for Top Narratives ---
    plot_top_narratives_performance(results_df, prices_df, final_summary, today_str)


if __name__ == '__main__':
    analyze_narrative_performance()




