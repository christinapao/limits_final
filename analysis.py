import pandas as pd
import argparse
import re

PRESENT_COUNTRIES = ["Viet Nam", "Russian Federation", "Cambodia", "Rwanda"]
NA_THRESHOLD = 0.9

def rotate_frame(df):
    yr_cols = [c for c in df.columns if re.search(r'\[YR', c) is not None]
    years = [c.split()[0] for c in yr_cols]
    series = dict()
    for _, row in df.iterrows():
        series_name = row['Series Name']
        series_code = row['Series Code']
        if series_code not in series:
            series[(series_code, series_name)] = {}
        for year, year_col in zip(years, yr_cols):
            series[(series_code, series_name)][year] = row[year_col]

    rotated = pd.DataFrame(series)
    rotated.index.name = 'Year'
    # print(rotated, rotated.index)
    # exit(1)
    return pd.DataFrame(series)


def filter_frame(df):
    present_df = df[df['Country Name'].isin(PRESENT_COUNTRIES)]

    keep_rows = set()
    yr_regex = r'\[YR'
    yr_cols = [c for c in present_df.columns if re.search(yr_regex, c) is not None]
    yr_count = len(yr_cols)
    for idx, row in present_df.iterrows():
        country = row['Country Name']
        na_count = row[yr_cols].isna().sum()
        if na_count >= yr_count * NA_THRESHOLD:
            continue
        keep_rows.add(idx)
    
    filtered_df = present_df[present_df.index.isin(keep_rows)]

    country_frames = [(country, filtered_df[filtered_df['Country Name'] == country].copy()) for country in PRESENT_COUNTRIES]
    return country_frames
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze some data.")
    parser.add_argument("-d", "--datafile", type=str, help="The path to the data file.")
    args = parser.parse_args()
    df = pd.read_csv(args.datafile)
    
    country_frames = filter_frame(df)
    for country, frame in country_frames:
        frame = rotate_frame(frame)
        snake_country = country.lower().replace(" ", "_")
        print(frame)
        frame.to_csv(f"{snake_country}.csv", index=True)
