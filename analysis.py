import pandas as pd
import argparse
import re
from tqdm import tqdm

PRESENT_COUNTRIES = ["Viet Nam", "Russian Federation", "Cambodia", "Rwanda"]
NA_THRESHOLD = 0.9

def rotate_frame(df):
    yr_cols = [c for c in df.columns if re.search(r'\[YR', c) is not None]
    years = [c.split()[0] for c in yr_cols]
    series = dict()

    for _, row in df.iterrows():
        series_name = row['Series Name']
        series_code = row['Series Code']
        country = row["Country Name"]
        code = row["Country Code"]

        series_key = f"{series_code}_{series_name}"

        if series_key not in series:
            series[series_key] = {}
        for year, year_col in zip(years, yr_cols):
            series[series_key][(country, year, code)] = row[year_col]

    rotated = pd.DataFrame(series)
    rotated.index.set_names(["Country", "Year", "Code"], inplace=True)
    print(rotated)
    # rotated.index.name = 'Year'
    # print(rotated, rotated.index)
    # exit(1)
    return pd.DataFrame(series)


def filter_frame(df):
    # present_df = df[df['Country Name'].isin(PRESENT_COUNTRIES)]
    present_df = df
    print("Before filtering, shape is", present_df.shape)
    # Drop all the rows where the country code is nan
    df = df[df['Country Code'].notna()].copy()

    keep_rows = set()
    yr_regex = r'\[YR'
    yr_cols = [c for c in present_df.columns if re.search(yr_regex, c) is not None]
    yr_count = len(yr_cols)
    for idx, row in tqdm(present_df.iterrows()):
        country =row['Country Name']
        na_count = row[yr_cols].isna().sum()
        if na_count >= yr_count * NA_THRESHOLD:
            continue
        keep_rows.add(idx)

    filtered_df = present_df[present_df.index.isin(keep_rows)]
    print("After filtering, shape is", filtered_df.shape)
    countries_present = len(filtered_df["Country Name"].unique())
    print("Remaing features ~=", len(filtered_df)/countries_present, "over", countries_present, "countries")
    return filtered_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze some data.")
    parser.add_argument("-d", "--datafile", type=str, help="The path to the data file.")
    parser.add_argument("--filtered", action="store_true", help="Is the data filtered already?")
    args = parser.parse_args()
    df = pd.read_csv(args.datafile)

    if not args.filtered:
        df = filter_frame(df)
        df.to_csv("filtered.csv", index=False)
    print(df)
    rotated = rotate_frame(df)
    rotated.to_csv("rotated.csv", index=True, index_label=["Country", "Year", "Code"])
    # for country, frame in country_frames:
    #     frame = rotate_frame(frame)
    #     snake_country = country.lower().replace(" ", "_")
    #     print(frame)
    #     frame.to_csv(f"{snake_country}.csv", index=True)
