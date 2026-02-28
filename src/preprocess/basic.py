import pandas as pd
import os


columns= (
    ["engine_number", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_raw(file_name: str, data_dir: str = "data/raw") -> pd.DataFrame:
    file_path = os.path.join(data_dir, file_name)
    
    df = pd.read_csv(
        file_path,
        sep=r'\s+',           # multiple spaces tabs
        header=None,          # no header in raw file
        names=columns,        # assign our column names
        engine='python'       # required for regex separator
    )
    
    print(f"Loaded {file_name}: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def drop_low_variance(df: pd.DataFrame, threshold: float = 0.001) -> pd.DataFrame:
    
    # Calculate std for all columns except IDs
    exclude_from_check = ["engine_number", "cycle"]
    cols_to_check = [col for col in df.columns if col not in exclude_from_check]
    
    std_values = df[cols_to_check].std()
    low_variance_cols = std_values[std_values < threshold].index.tolist()
    
    if low_variance_cols:
        print(f"📉 Dropping {len(low_variance_cols)} low-variance columns (std < {threshold}):")
        for col in low_variance_cols:
            print(f"   - {col} (std = {df[col].std():.6f})")
        df = df.drop(columns=low_variance_cols)
    else:
        print(f"✅ No columns dropped (all have std >= {threshold})")
    
    return df


def add_rul_and_clip(df: pd.DataFrame, max_rul: int = 130) -> pd.DataFrame:
   
    #Find max cycle for each engine (failure time)
    max_cycle_per_engine = df.groupby("engine_number")["cycle"].max().reset_index()
    max_cycle_per_engine.columns = ["engine_number", "max_cycle"]
    
    #Merge back to original dataframe
    df = df.merge(max_cycle_per_engine, on="engine_number", how="left")
    
    #Calculate RUL
    df["RUL"] = df["max_cycle"] - df["cycle"]
    
    #Create capped version (target variable for modeling)
    df["RUL_capped"] = df["RUL"].clip(upper=max_rul)
    
    #Drop temporary max_cycle column
    df = df.drop(columns=["max_cycle", "RUL"])
    
    # Safe printing – only use columns that still exist
    print(f"✅ Added RUL columns:")
    print(f"   - RUL_capped: clipped at {max_rul} cycles")
    
    # Count how many rows hit the cap (using RUL_capped)
    capped_count = (df["RUL_capped"] == max_rul).sum()
    print(f"   - {capped_count:,} rows capped "
          f"({capped_count / len(df) * 100:.1f}% of rows)")
    
    return df


def basic_preprocess_train(
    file_name: str = "train_FD001.txt",
    variance_threshold: float = 0.001,
    max_rul: int = 130,
    save_path: str = "data/processed/train_clean.csv"
    
) -> pd.DataFrame:
    
    print("PREPROCESSING TRAINING DATA")

    
    # Step 1: Load
    df = load_raw(file_name)
    
    # Step 2: Drop constants
    df = drop_low_variance(df, threshold=variance_threshold)
    
    # Step 3: Add RUL
    df = add_rul_and_clip(df, max_rul=max_rul)
    
    # Step 4: Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"💾 Saved cleaned training data to: {save_path}")
    print(f"   Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    
    return df


def basic_preprocess_test(
    file_name: str = "test_FD001.txt",
    variance_threshold: float = 0.001,
    save_path: str = "data/processed/test_clean.csv"
) -> pd.DataFrame:
    

    print("PREPROCESSING TEST DATA")

    
    # Step 1: Load
    df = load_raw(file_name)
    
    # Step 2: Drop same constants as training
    df = drop_low_variance(df, threshold=variance_threshold)
    
    # NO RUL calculation for test data
    print("ℹ️  Skipping RUL calculation (test data has unknown failure times)")
    
    # Step 3: Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"💾 Saved cleaned test data to: {save_path}")
    print(f"   Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    
    return df


if __name__ == "__main__":
    # This code runs if you execute the module directly
    # Useful for testing
    print("Testing preprocessing module...\n")
    
    # Test training preprocessing
    df_train = basic_preprocess_train()
    print(f"\nTrain columns: {df_train.columns.tolist()}")
    
    # Test test preprocessing
    df_test = basic_preprocess_test()
    print(f"\nTest columns: {df_test.columns.tolist()}")
    
    print("\n✅ Module test complete!")