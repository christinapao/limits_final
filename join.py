import argparse
import pandas as pd

to_geo = lambda x : f"W140000WO{x.upper()}"

IDP_COLS = ["#YR", "POP", "NMR", "DEATHS", "BIRTHS"]

def join_idb(df, idb, code):
    country_idb = idb[idb['GEO_ID'] == to_geo(code)]
    latest_year = df.index.max()
    earliest_year = df.index.min()
    # Filter out the years so that we match on both dataframes
    country_idb = country_idb[(country_idb['#YR'] <= latest_year) & (country_idb['#YR'] >= earliest_year)]
    country_idb = country_idb[IDP_COLS]
    country_idb.set_index("#YR", inplace=True)
    # Join on the year column
    print("BEFORE MERGE", df)
    merged = df.reset_index().merge(country_idb, how='outer', left_on="Year", right_on="#YR").set_index("Year")
    new_cols = {}
    for col in merged:
        if type(col) == tuple:
            new_cols[col] = col[1]
    merged.rename(columns=new_cols, inplace=True)
    return merged

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

def fix_country_df(df):
    new_cols = {}
    for idx, col in enumerate(df.columns):
        series_name = df[col].iloc[0]
        new_cols[col] = (col, series_name)
    df.rename(columns=new_cols, inplace=True)
    df = df.iloc[1:].copy()
    df.index.name = "Year"
    df.index = df.index.astype(int)
    return df

if __name__ == "__main__":
    print("=== looking over the world bank data ===")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file")
    parser.add_argument('--idb', help="idb file")
    parser.add_argument('--code', help="country code to use")
    args = parser.parse_args()

    df = pd.read_csv(args.input, index_col=0)
    df = fix_country_df(df)
    df = filter_country_df(df)
    idb_df = pd.read_csv(args.idb, delimiter='|')
    joined_data = join_idb(df, idb_df, args.code)
    print(joined_data) 
    joined_data.to_csv(f"joined_data/{args.code}_joined.csv", index=True)