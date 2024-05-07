import argparse
import pandas as pd

to_geo = lambda x : f"W140000WO{x.upper()}"

def get_decline(df, start, end):
    values = []
    for i in range(start, end):
        yr_row = df[df["#YR"] == i]
        pop = yr_row["POP"].values[0]
        values.append((i, pop))

    delta = [(values[i][0], values[i][1] - values[i-1][1]) for i in range(1, len(values))]
    print(values)
    print(delta)
    total = sum([x[1] for x in delta])
    print(total)
    pct_decrease = total / values[0][1]
    print("Decrease by", pct_decrease * 100, "%")
    return total

def get_total_growth(df, start, end):
    # Sum up all the Pops in end and start and compare
    start_pop = df[df["#YR"] == start]["POP"].sum()
    end_pop = df[df["#YR"] == end]["POP"].sum()
    print("Start population", start_pop)
    print("End population", end_pop)
    print("Total growth", end_pop - start_pop)
             

if __name__ == "__main__":
    print("=== looking over the world bank data ===")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="The world bank data")
    parser.add_argument("--code", type=str, help="Code of the country to use")
    parser.add_argument("--start", type=int, help="Start year")
    parser.add_argument("--end", type=int, help="End year")
    args = parser.parse_args()

    # Load the data
    df = pd.read_parquet(args.data)
    get_total_growth(df, args.start, args.end)
    df = df[df["GEO_ID"] == to_geo(args.code)]
    get_decline(df, args.start, args.end)