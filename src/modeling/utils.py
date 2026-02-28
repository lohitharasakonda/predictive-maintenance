"""
Modeling utilities for NASA C-MAPSS predictive maintenance.

This module provides:
- Feature scaling (StandardScaler)
- Train/validation splitting by engine
- X/y extraction
- Model training and evaluation helpers
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os


def scale_features(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    is_fit: bool = False
) -> tuple:
    """
    Scale features using StandardScaler.
    
    CRITICAL: Always fit scaler on TRAINING data only, then transform train/val/test.
    
    Columns excluded from scaling:
    - engine_number, cycle (identifiers)
    - RUL, RUL_capped (targets)
    
    Args:
        df: Input DataFrame
        scaler: Existing scaler (if transforming val/test)
        is_fit: If True, create and fit new scaler on this data
                If False, use provided scaler to transform
    
    Returns:
        Tuple of (scaled_df, scaler)
        - If is_fit=True: returns fitted scaler
        - If is_fit=False: returns None for scaler (since it was provided)
    """
    df = df.copy()
    
    # Identify columns to exclude from scaling
    exclude_cols = ['engine_number', 'cycle', 'RUL_capped']
    
    # Get feature columns (everything except excluded)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if is_fit:
        # Create and fit scaler on training data
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        print(f"✅ Fitted scaler on {len(feature_cols)} feature columns")
        print(f"   Mean (before scaling): {df[feature_cols].mean().mean():.4f}")
        print(f"   Std (before scaling):  {df[feature_cols].std().mean():.4f}")
        print(f"   After scaling: mean ≈ 0, std ≈ 1")
        return df, scaler
    else:
        # Transform using existing scaler
        if scaler is None:
            raise ValueError("Must provide scaler when is_fit=False")
        df[feature_cols] = scaler.transform(df[feature_cols])
        print(f"✅ Transformed {len(feature_cols)} feature columns using provided scaler")
        return df, None


def split_by_engines(
    df: pd.DataFrame,
    train_engines: int = 70,
    val_engines: int = 15
) -> tuple:
    """
    Split data by engine IDs (not random rows).
    
    Why? Because cycles from same engine are correlated.
    Random split would leak information from validation engines into training.
    
    Args:
        df: Input DataFrame (must have 'engine_number' column)
        train_engines: Number of engines for training
        val_engines: Number of engines for validation
    
    Returns:
        Tuple of (df_train_split, df_val_split)
    """
    # Get unique engine IDs and sort
    unique_engines = sorted(df['engine_number'].unique())
    
    if len(unique_engines) < train_engines + val_engines:
        raise ValueError(
            f"Not enough engines! Have {len(unique_engines)}, "
            f"need {train_engines + val_engines}"
        )
    
    # Split engine IDs
    train_engine_ids = unique_engines[:train_engines]
    val_engine_ids = unique_engines[train_engines:train_engines + val_engines]
    
    # Filter dataframe
    df_train_split = df[df['engine_number'].isin(train_engine_ids)].copy()
    df_val_split = df[df['engine_number'].isin(val_engine_ids)].copy()
    

    print("TRAIN/VALIDATION SPLIT BY ENGINES")

    print(f"Total engines available: {len(unique_engines)}")
    print(f"\nTrain split:")
    print(f"  Engines: {train_engines} (IDs: {train_engine_ids[0]}-{train_engine_ids[-1]})")
    print(f"  Rows: {len(df_train_split):,}")
    print(f"\nValidation split:")
    print(f"  Engines: {val_engines} (IDs: {val_engine_ids[0]}-{val_engine_ids[-1]})")
    print(f"  Rows: {len(df_val_split):,}")
    print(f"\nHeld-out engines: {len(unique_engines) - train_engines - val_engines}")

    
    return df_train_split, df_val_split


def get_xy(df: pd.DataFrame) -> tuple:
    """
    Separate features (X) from target (y).
    
    X = all columns except: engine_number, cycle, RUL, RUL_capped
    y = RUL_capped (our target variable)
    
    Args:
        df: Input DataFrame
    
    Returns:
        Tuple of (X, y)
        - X: DataFrame of features
        - y: Series of target values (or None if no target in df)
    """
    # Columns to exclude from features
    exclude_from_X = ['engine_number', 'cycle','RUL_capped']
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_from_X]
    X = df[feature_cols].copy()
    
    # Get target if it exists
    if 'RUL_capped' in df.columns:
        y = df['RUL_capped'].copy()
    else:
        y = None
        print("ℹ️  No target column (RUL_capped) found - returning X only")
    
    print(f"📊 Extracted features and target:")
    print(f"   X shape: {X.shape}")
    if y is not None:
        print(f"   y shape: {y.shape}")
        print(f"   y range: [{y.min():.1f}, {y.max():.1f}]")
    
    return X, y


def train_and_evaluate(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str
) -> dict:
    """
    Train a model and evaluate on validation set.
    
    Args:
        model: Sklearn model instance (must have .fit() and .predict())
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        model_name: Name for display purposes
    
    Returns:
        Dictionary with metrics: {'rmse': float, 'mae': float, 'r2': float}
    """

    print(f"TRAINING: {model_name}")
    
    # Train
    print("🔧 Fitting model...")
    model.fit(X_train, y_train)
    print("✅ Model fitted!")
    
    # Predict on validation
    y_pred_val = model.predict(X_val)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae = mean_absolute_error(y_val, y_pred_val)
    r2 = r2_score(y_val, y_pred_val)
    
    print(f"\n📊 Validation Results:")
    print(f"   RMSE: {rmse:.2f} cycles")
    print(f"   MAE:  {mae:.2f} cycles")
    print(f"   R²:   {r2:.4f}")

    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred_val
    }


def save_model(model, filename: str, models_dir: str = "models"):
    """
    Save a trained model using joblib.
    
    Args:
        model: Trained sklearn model
        filename: Name of file (e.g., 'rf_model.pkl')
        models_dir: Directory to save to
    """
    os.makedirs(models_dir, exist_ok=True)
    filepath = os.path.join(models_dir, filename)
    joblib.dump(model, filepath)
    print(f"💾 Saved model to: {filepath}")


def load_model(filename: str, models_dir: str = "models"):
    """
    Load a saved model.
    
    Args:
        filename: Name of file (e.g., 'rf_model.pkl')
        models_dir: Directory to load from
    
    Returns:
        Loaded sklearn model
    """
    filepath = os.path.join(models_dir, filename)
    model = joblib.load(filepath)
    print(f"📂 Loaded model from: {filepath}")
    return model


if __name__ == "__main__":
    # Test the module
    print("Testing modeling utils module...\n")
    
    # Load featured data
    if os.path.exists("data/processed/train_features.csv"):
        df = pd.read_csv("data/processed/train_features.csv")
        print(f"Loaded train_features.csv: {df.shape}")
        
        # Test scaling
        df_scaled, scaler = scale_features(df, is_fit=True)
        
        # Test splitting
        df_train, df_val = split_by_engines(df_scaled, train_engines=70, val_engines=15)
        
        # Test X/y extraction
        X_train, y_train = get_xy(df_train)
        X_val, y_val = get_xy(df_val)
        
        print("\n✅ Module test complete!")
    else:
        print("❌ train_features.csv not found. Run feature engineering first!")