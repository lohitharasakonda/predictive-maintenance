"""
Time-series feature engineering for NASA C-MAPSS dataset.

This module creates temporal features that capture sensor degradation patterns:
- Rolling statistics (mean, std) to capture recent trends
- Lag features to capture past values
"""

import pandas as pd
import numpy as np


def add_rolling_and_lags(
    df: pd.DataFrame,
    windows: list = [5, 10],
    lags: list = [1, 3],
    drop_originals: bool = True
) -> pd.DataFrame:
    """
    Add rolling statistics and lag features for all sensors.
    
    Rolling features capture recent trends (smoothed values, volatility).
    Lag features capture past values (how quickly things change).
    """
    print("FEATURE ENGINEERING: Rolling Statistics & Lags")
    
    df = df.copy()
    df = df.sort_values(['engine_number', 'cycle']).reset_index(drop=True)
    original_rows = len(df)
    
    # Identify sensor columns (exclude IDs, settings, and targets)
    sensor_cols = [col for col in df.columns if 'sensor' in col]
    print(f"📊 Processing {len(sensor_cols)} sensor columns")
    print(f"   Sensors: {sensor_cols}")
    
    
    new_features = []
    
    
    # ROLLING STATISTICS
    print(f"\n🔄 Adding rolling statistics (windows: {windows})...")
    for window in windows:
        for sensor in sensor_cols:
            col_name_mean = f"{sensor}_mean_{window}"
            df[col_name_mean] = df.groupby('engine_number')[sensor].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            new_features.append(col_name_mean)
            
            col_name_std = f"{sensor}_std_{window}"
            df[col_name_std] = df.groupby('engine_number')[sensor].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            new_features.append(col_name_std)
    
    print(f"   ✅ Added {len(windows) * len(sensor_cols) * 2:,} rolling features "
          f"({len(windows)} windows × {len(sensor_cols)} sensors × 2)")
    
    # LAG FEATURES
    print(f"\n⏮️ Adding lag features (lags: {lags})...")
    for lag in lags:
        for sensor in sensor_cols:
            col_name_lag = f"{sensor}_lag_{lag}"
            df[col_name_lag] = df.groupby('engine_number')[sensor].shift(lag)
            new_features.append(col_name_lag)
    
    print(f"   ✅ Added {len(lags) * len(sensor_cols):,} lag features "
          f"({len(lags)} lags × {len(sensor_cols)} sensors)")
    
    # HANDLE NaN VALUES
    print(f"\n🧹 Handling NaN values...")
    nan_before = df.isnull().sum().sum()
    print(f"   NaN count before cleaning: {nan_before:,}")
    
    df = df.dropna().reset_index(drop=True)
    
    rows_after = len(df)
    rows_dropped = original_rows - rows_after
    nan_after = df.isnull().sum().sum()
    
    print(f"   NaN count after cleaning: {nan_after:,}")
    print(f"   Rows dropped due to NaN: {rows_dropped:,} "
          f"({rows_dropped / original_rows * 100:.1f}% of original)")
    
    # DROP ORIGINAL SENSORS 
    if drop_originals:
        print(f"\n🗑️ Dropping original sensor columns (keeping engineered features)...")
        df = df.drop(columns=sensor_cols)
        print(f"   Dropped {len(sensor_cols)} original sensor columns")
    
    # Summary 
    print(f"\n📈 Feature Engineering Summary:")
    print(f"   Original sensors processed: {len(sensor_cols)}")
    print(f"   Rolling features added: {len(windows) * len(sensor_cols) * 2:,}")
    print(f"   Lag features added: {len(lags) * len(sensor_cols):,}")
    print(f"   Total new features added: {len(new_features):,}")
    
    # Safe count of modeling features
    exclude_cols = ['engine_number', 'cycle']
    if 'RUL_capped' in df.columns:
        exclude_cols.append('RUL_capped')
    
    modeling_features = len(df.columns) - len(exclude_cols)
    print(f"   Total modeling features (excluding IDs & target): {modeling_features:,}")
    print(f"   Final shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    
    return df


def add_rolling_only(
    df: pd.DataFrame,
    windows: list = [5, 10]
) -> pd.DataFrame:
    return add_rolling_and_lags(df, windows=windows, lags=[], drop_originals=False)


def add_lags_only(
    df: pd.DataFrame,
    lags: list = [1, 3]
) -> pd.DataFrame:
    return add_rolling_and_lags(df, windows=[], lags=lags, drop_originals=False)


if __name__ == "__main__":
    # Test the module
    print("Testing feature engineering module...\n")
    
    # Load cleaned data
    import os
    if os.path.exists("data/processed/train_clean.csv"):
        df_train = pd.read_csv("data/processed/train_clean.csv")
        print(f"Loaded train_clean.csv: {df_train.shape}")
        
        # Apply feature engineering
        df_train_features = add_rolling_and_lags(
            df_train,
            windows=[5, 10],
            lags=[1, 3],
            drop_originals=True
        )
        
        print(f"\nResult shape: {df_train_features.shape}")
        print(f"Columns: {df_train_features.columns.tolist()}")
        
        print("\n✅ Module test complete!")
    else:
        print("❌ train_clean.csv not found. Run preprocessing first!")