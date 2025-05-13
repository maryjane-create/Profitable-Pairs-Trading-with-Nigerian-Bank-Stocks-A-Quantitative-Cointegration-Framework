import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

def load_close_series(filepath, ticker_name):
    """
    Load CSV data and extract the closing price with date as index.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=['time'])
    df = df[['time', 'close']].dropna()
    df.rename(columns={'close': f'{ticker_name}_close'}, inplace=True)
    df.set_index('time', inplace=True)
    df = df['2015-01-01':'2025-12-31']
    return df

def determine_pacesetter(file1_path, file2_path, ticker1_name, ticker2_name):
    """
    Determine the pacesetter stock by comparing residual standard deviations
    from two regressions: ticker1 ~ ticker2 and ticker2 ~ ticker1.
    """
    # Load data
    df1 = load_close_series(file1_path, ticker1_name)
    df2 = load_close_series(file2_path, ticker2_name)

    # Merge dataframes
    df = pd.merge(df1, df2, left_index=True, right_index=True).dropna()

    # Regression 1: ticker1 ~ ticker2
    X1 = sm.add_constant(df[f'{ticker2_name}_close'])
    model1 = sm.OLS(df[f'{ticker1_name}_close'], X1).fit()
    resid_std1 = np.std(model1.resid)

    # Regression 2: ticker2 ~ ticker1
    X2 = sm.add_constant(df[f'{ticker1_name}_close'])
    model2 = sm.OLS(df[f'{ticker2_name}_close'], X2).fit()
    resid_std2 = np.std(model2.resid)

    # Print results
    print("\n--- Residual Standard Deviations ---")
    print(f"{ticker1_name} ~ {ticker2_name}: {resid_std1:.4f}")
    print(f"{ticker2_name} ~ {ticker1_name}: {resid_std2:.4f}")

    # Determine pacesetter
    if resid_std1 < resid_std2:
        print(f"\n✅ {ticker2_name} is the pacesetter (lower residual std).")
        return ticker2_name
    else:
        print(f"\n✅ {ticker1_name} is the pacesetter (lower residual std).")
        return ticker1_name

# Example usage:
if __name__ == "__main__":
    pacesetter = determine_pacesetter(
        "/content/NSENG_FBNH, 1D_15c8b.csv",
        "/content/NSENG_ACCESSCORP, 1D_71fee.csv",
        "FBNH", "ACCESSBANK"
    )
