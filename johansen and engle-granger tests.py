import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import os

def load_and_prepare_data(filepath, ticker_name):
    """
    Load a CSV file containing OHLCV data, extract the 'Date' and 'Close' columns,
    set the date as index, rename the 'Close' column to include ticker_name, and filter date range.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=['time'])
    df = df[['time', 'close']].dropna()
    df = df.rename(columns={'close': f'{ticker_name}_close'})
    df.set_index('time', inplace=True)
    df = df['2015-01-01':'2025-12-31']
    return df

def perform_engle_granger_test(series1, series2):
    """
    Perform Engle-Granger 2-step cointegration test.
    """
    print("\n--- Engle-Granger Two-Step Cointegration Test ---")
    score, pvalue, _ = coint(series1, series2)
    print(f"Test Statistic: {score:.4f}")
    print(f"P-value: {pvalue:.4f}")
    if pvalue < 0.05:
        print("✅ Series are likely cointegrated (reject null hypothesis of no cointegration).")
    else:
        print("❌ No evidence of cointegration (fail to reject null).")

def perform_johansen_test(df):
    """
    Perform Johansen Cointegration Test.
    """
    print("\n--- Johansen Cointegration Test ---")
    johansen = coint_johansen(df, det_order=0, k_ar_diff=1)

    trace_stat = johansen.lr1
    crit_vals = johansen.cvt

    for i in range(len(trace_stat)):
        print(f"\nRank {i}:")
        print(f"Trace Statistic: {trace_stat[i]:.4f}")
        print(f"Critical Values (90%, 95%, 99%): {crit_vals[i]}")
        if trace_stat[i] > crit_vals[i][1]:
            print("✅ Evidence of cointegration at 5% level.")
        else:
            print("❌ No cointegration at 5% level.")

def plot_series(df):
    """
    Plot the two closing price series.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df.iloc[:, 0], label=df.columns[0])
    plt.plot(df.index, df.iloc[:, 1], label=df.columns[1])
    plt.title('UBA vs ZENITHBANK Closing Prices (2015–2025)', fontsize=14)
    plt.xlabel('time')
    plt.ylabel('Closing Price (NGN)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_cointegration_pipeline(uba_file_path, gtco_file_path):
    """
    Load data, perform plotting and both cointegration tests.
    """
    # Load data
    uba = load_and_prepare_data(uba_file_path, 'UBA')
    gtco = load_and_prepare_data(gtco_file_path, 'ZENITHBANK')

    # Merge datasets on Date
    df_combined = pd.merge(uba, gtco, left_index=True, right_index=True).dropna()
    df_combined = np.log(df_combined)

    # Plot time series
    plot_series(df_combined)

    # Perform Engle-Granger Test
    perform_engle_granger_test(df_combined['UBA_close'], df_combined['ZENITHBANK_close'])

    # Perform Johansen Test
    perform_johansen_test(df_combined)

def main():
    """
    Main entry point of the script.
    Replace the file paths with the actual location of your UBA and GTCO CSV files.
    """
    uba_file = "/content/NSENG_UBA, 1D_4a819.csv"     # Replace with your actual path
    gtco_file = "/content/NSENG_ZENITHBANK, 1D_6d9c1.csv"   # Replace with your actual path

    try:
        run_cointegration_pipeline(uba_file, gtco_file)
    except Exception as e:
        print(f"Error: {e}")

# This ensures main() only runs when the script is directly executed
if __name__ == "__main__":
    main()
