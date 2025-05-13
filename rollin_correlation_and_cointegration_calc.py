import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import os

# Step 1: Read 20 user-specified files containing 'Date' and 'Close' columns
# Define timeframes
timeframes = {
    '2-year': ('2023-03-01', '2025-03-01'),
    '5-year': ('2020-01-01', '2025-01-01'),
    '8-year': ('2017-01-01', '2025-01-01'),
    '10-year': ('2015-01-01', '2025-01-01')
}

# Example file paths – REPLACE these with your actual file paths fetched from TradingView or any other source
file_paths = {
    'GTCO': '/content/NSENG_GTCO, 1D_e2363.csv',
    'UBA': '/content/NSENG_UBA, 1D_d298c.csv',
    'ZENITHBANK': '/content/NSENG_ZENITHBANK, 1D_6d9c1.csv',
    'MTNN': '/content/NSENG_MTNN, 1D_ed527.csv',
    'AIRTELAFRICA': '/content/NSENG_AIRTELAFRI, 1D_1e3ed.csv',
    'DANGCEM': '/content/NSENG_DANGCEM, 1D_bbb2a.csv',
    'BUACEMENT': '/content/NSENG_BUACEMENT, 1D_9928b.csv',
    'UNILEVER': '/content/NSENG_UNILEVER, 1D_255fa.csv',
    'NESTLE': '/content/NSENG_NESTLE, 1D_7b085.csv',
    'FBN': '/content/NSENG_FBNH, 1D_15c8b.csv',
    'FCMB': '/content/NSENG_FCMB, 1D_046f3.csv',
    'ACCESSBANK': '/content/NSENG_ACCESSCORP, 1D_71fee.csv',
    'FIDELITYBANK': '/content/NSENG_FIDELITYBK, 1D_11cce.csv',
    'ETI': '/content/NSENG_ETI, 1D_26f95.csv',
    'GUINESS': '/content/NSENG_GUINNESS, 1D_f13f7.csv',
    'FLOURMILLS': '/content/NSENG_FLOURMILL, 1D_ad9dd.csv',
    'CONOIL': '/content/NSENG_CONOIL, 1D_61bad.csv',
    'OANDO': '/content/NSENG_OANDO, 1D_b0fdc.csv',
    'SEPLAT': '/content/NSENG_SEPLAT, 1D_b2172.csv',
    'MAYANDBAKER': '/content/NSENG_MAYBAKER, 1D_bb971.csv',
    'JULIUSBERGER': '/content/NSENG_JBERGER, 1D_20fac.csv',
    'JOHNHOLT': '/content/NSENG_JOHNHOLT, 1D_a5aa9.csv',
    'UAC': '/content/NSENG_UACN, 1D_6163f.csv',
    'TRANSCORP': '/content/NSENG_TRANSCORP, 1D_62ab7.csv',
    'UPDC': '/content/NSENG_UPDC, 1D_58c00.csv',
    'IKEJAHOTEL': '/content/NSENG_IKEJAHOTEL, 1D_8db9f.csv',
    'ETERNA': '/content/NSENG_ETERNA, 1D_d1021.csv',
    'OMATEK': '/content/NSENG_OMATEK, 1D_b5529.csv',
    'PZ': '/content/NSENG_PZ, 1D_a1fb5.csv',
    'STANBICIBTC': '/content/NSENG_STANBIC, 1D_da76f.csv',
    'TOTALENERGIES': '/content/NSENG_TOTAL, 1D_9ee25.csv'
}

# Step 2: Load and merge all data into a single DataFrame
def load_close_prices(file_paths):
    all_data = []
    for ticker, path in file_paths.items():
        try:
            # Check if the file exists before attempting to read it
            if os.path.exists(path):
                df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                df = df[['close']].rename(columns={'close': ticker})
                all_data.append(df)
            else:
                print(f"File not found: {path} for ticker {ticker}")  # Print a warning if the file is not found

        except Exception as e:
            print(f"Error reading {ticker}: {e}")

    # Check if all_data is empty before concatenating
    if all_data:
        merged = pd.concat(all_data, axis=1)
        merged.sort_index(inplace=True)
        return merged.dropna(axis=1, how='all')  # Drop tickers with all missing values
    else:
        print("No data loaded from any files. Check file paths and formats.")
        return pd.DataFrame()  # Return an empty DataFrame if no data is loaded


price_data = load_close_prices(file_paths)

# Convert daily prices to monthly prices
def resample_to_monthly(data):
    """
    Resamples daily data to monthly data using the last trading day's closing price.
    """
    return data.resample('3M').last()

monthly_price_data = resample_to_monthly(price_data)

# Print loaded tickers
print("\nLoaded Tickers:")
print(list(monthly_price_data.columns))

# Step 3: Calculate Correlation Table
def calculate_correlation(data, start_date, end_date):
    data_in_timeframe = data[(data.index >= start_date) & (data.index <= end_date)]
    return data_in_timeframe.corr()


correlation_tables = {}
for name, (start, end) in timeframes.items():
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    correlation_tables[name] = calculate_correlation(monthly_price_data, start_date, end_date)

    # Create a separate Excel file for each time interval
    output_filename = f"{name}_correlation_table.xlsx"
    correlation_tables[name].to_excel(output_filename)
    print(f"\n{name} Correlation Table (saved to {output_filename}):")
    print(correlation_tables[name])

# Step 4: Cointegration Tests
def perform_cointegration_tests(data):
    n = len(data.columns)
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            asset1 = data.columns[i]
            asset2 = data.columns[j]
            try:
                score, pvalue, _ = coint(data[asset1], data[asset2])
                results.append((asset1, asset2, score, pvalue))
            except Exception as e:
                print(f"Error performing cointegration test for {asset1} and {asset2}: {e}")
    return pd.DataFrame(results, columns=['Asset1', 'Asset2', 'Cointegration Score', 'P-Value'])


cointegration_results = perform_cointegration_tests(monthly_price_data)

print("\n=== Cointegration Results (p-value < 0.05) ===")
significant = cointegration_results[cointegration_results['P-Value'] < 0.05]
if not significant.empty:
    print(significant)
    significant.to_csv("cointegration_results.csv", index=False)
else:
    print("No significant cointegrated pairs found.")