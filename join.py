import argparse
import pandas as pd
import numpy as np


SCAR_COUNTRIES = ["RU", "KH", "RW"]

col_to_code = lambda col : col.split("_")[0]


def join_idb(df, idb):
    """Join the idb data with the world bank data"""
    min_year = df["Year"].min()
    max_year = df["Year"].max()
    idb_df = idb[(idb["#YR"] >= min_year) & (idb["#YR"] <= max_year)]
    # Now we should just be able to join on (year, code) pairing
    print("World bank shape:", df.shape)
    merged_df = df.reset_index().merge(idb_df, left_on=["Year", "Code"], right_on=["#YR", "GEO_ID"], how="inner").set_index(["Country", "Year", "Code"])
    merged_df.drop(columns=["#YR", "GEO_ID"], inplace=True)
    # We lose some rows in the merge, but that's probably fine. Mostly likely due to mismatch in country codes
    print("Merged shape:", merged_df.shape)
    return merged_df

def filter_country_df(df):
    col_orders = []
    keep_cols = []
    for code, name in df.columns:
        if "estimate" in name.lower():
            continue
        if "SP.POP" in code:
            keep_cols.append((code, name))
    # print(keep_cols)
    df = df[keep_cols]
    # print(df)
    return df

def fix_worldbank_df(df, codes):
    """Fix the world bank data to use iso2 codes"""
    # Just construct a dictionary because it's faster
    three_to_two = dict()
    for _, row in codes.iterrows():
        three_to_two[row["alpha-3"]] = row["alpha-2"]
    # Remove everything that's not in the dictionary because finding a match is futile
    df = df[df["Code"].isin(three_to_two)].copy()
    # Remove columns with high nan percentage
    drop_cols = []
    for col in df.columns:
        na_count = df[col].isna().sum()
        if na_count >= len(df) * 0.9:
            drop_cols.append(col)
    df.drop(columns=drop_cols, inplace=True)
    # Swap all the codes to the format used by the idb data
    df["Code"] = df["Code"].apply(lambda x : f"W140000WO{three_to_two[x].upper()}")
    return df


def normalize_col(col, df: pd.DataFrame):
    col_max, col_min = df[col].max(), df[col].min()
    code = col.split("_")[0]
    if code == "POP":
        print("Population constants:", col_max, col_min, "col", col)
    def normalize_value(x):
        if not pd.notna(x):
            return x
        return (x-col_min)/(col_max-col_min)

    if col_max == col_min:
        # If all the values are the same, set to 0
        df[col] = df[col].apply(lambda x: 0)
    else:
        df[col] =  df[col].apply(normalize_value)

def normalize_output(df):
    """take the joined dataframe (idb + world bank) and normalize the columns"""
    # All the remaining columns should be numbers
    for col in df.columns:
        normalize_col(col, df)
    return df

if __name__ == "__main__":
    print("=== looking over the world bank data ===")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file")
    parser.add_argument('--idb', help="idb file")
    args = parser.parse_args()

    # Don't filter nans because Namibia's two letter code is NA...
    codes_df = pd.read_csv("codes.csv", na_filter=False)
    print(codes_df)

    df = pd.read_csv(args.input, index_col=0, na_values=[".."])
    df = fix_worldbank_df(df, codes_df)

    idb_df = None
    if args.idb.endswith("txt"):
        idb_df = pd.read_csv(args.idb, delimiter='|')
        idb_df.to_parquet(args.idb.replace("txt", "parquet"))
    # Parquet files are a bit faster, let's prefer them
    if args.idb.endswith("parquet"):
        idb_df = pd.read_parquet(args.idb)
    merged_df = join_idb(df, idb_df)
    normalize_output(merged_df)
    print(merged_df)
    merged_df.to_csv("joined_data.csv", index=True, index_label=["Country", "Year", "Code"])
