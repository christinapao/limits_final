import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pickle

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from sklearn.model_selection import TimeSeriesSplit
tss = TimeSeriesSplit(n_splits=3)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

scar_ranges = {
    "W140000WOKH": (1974, 1979),
    "W140000WORU": (1997, 2010),
    "W140000WORW": (1992, 1997),
}

def denormalize_pop(pop):
    return pop * (1413142846 - 2705) + 2705

def add_previous_years(df):
    target_map = df['POP'].to_dict()
    df['lag1'] = (df.index - 1).map(target_map)
    df['lag2'] = (df.index - 2).map(target_map)
    df['lag3'] = (df.index - 3).map(target_map)
    df["lag1"] = df["lag1"].fillna(0)
    df["lag2"] = df["lag2"].fillna(0)
    df["lag3"] = df["lag3"].fillna(0)
    return df


def add_next_years(df):
    target_map = df['POP'].to_dict()
    df['lag1'] = (df.index + 1).map(target_map)
    df['lag2'] = (df.index + 2).map(target_map)
    df['lag3'] = (df.index + 3).map(target_map)
    df["goal"] = (df.index - 1).map(target_map)
    df["lag1"] = df["lag1"].fillna(0)
    df["lag2"] = df["lag2"].fillna(0)
    df["lag3"] = df["lag3"].fillna(0)
    df["goal"] = df["goal"].fillna(0)
    return df

def fill_na_nn(df):
    # Impute the missing values using nearest neighbors (this is slow, avoid for now)
    imputer = KNNImputer(n_neighbors=5)
    number_cols = list()
    non_numbers = list()
    for col in df.columns:
        if df[col].dtype == "float64" or df[col].dtype == "int64":
            number_cols.append(col)
        else:
            non_numbers.append(col)
    to_impute = df[number_cols].copy()
    print("=== Imputing ===")
    to_impute[:] = imputer.fit_transform(to_impute)
    print("=== Done imputing ===")
    # Add the non-number cols
    to_impute[non_numbers] = df[non_numbers]
    to_impute.set_index("Country", inplace=True)
    print(to_impute)
    
def fill_na_simple(df):
    for col in df.columns:
        if df[col].dtype == "float64" or df[col].dtype == "int64":
            imputer = SimpleImputer(strategy="mean")
            df[col] = imputer.fit_transform(df[col].values.reshape(-1, 1)) 
    return df

def forward_predict(df, country_df):
    """perform training and forward prediction for all three kinds of models"""
    target = 'POP'
    x_train, y_train = df.drop(target, axis=1), df[target]
    x_test, y_test = country_df.drop(target, axis=1), country_df[target]
    start_idx = y_test.index[0]
    start_year = 1960
    idx_fix = start_idx - start_year
    idx_fixer = lambda x : x - idx_fix
     
    results = {}
    models = [
        ("Decision tree", DecisionTreeRegressor(random_state=0)),
        ("Gradient boost", GradientBoostingRegressor(n_estimators=10)),
        ("Random forest", RandomForestRegressor(n_estimators=10)) 
    ]
    for name, model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test) 
        graph_pred = denormalize_pop(y_pred)
        results[name] = graph_pred

    graph_actual = denormalize_pop(y_test)
    results["Actual"] = graph_actual
    results["Index"] = [idx_fixer(x) for x in y_test.index]
    
    return results

def back_predict(df, country_df):
    """perform training and back prediction for all three kinds of models"""
    target = 'goal'
    x_train, y_train = df.drop(target, axis=1), df[target]
    x_test, y_test = country_df.drop(target, axis=1), country_df[target]
    start_idx = y_test.index[0]
    start_year = 1960
    idx_fix = start_idx - start_year
    idx_fixer = lambda x : x - idx_fix

    results = {}
    models = [
        ("Decision tree", DecisionTreeRegressor(random_state=0)),
        ("Gradient boost", GradientBoostingRegressor(n_estimators=10)),
        ("Random forest", RandomForestRegressor(n_estimators=10)) 
    ]
    for name, model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test) 
        graph_pred = denormalize_pop(y_pred)
        results[name] = graph_pred

    graph_actual = denormalize_pop(y_test)
    results["Actual"] = graph_actual
    results["Index"] = [idx_fixer(x) for x in y_test.index]
    
    return results

def fix_df(df, apply_func):
    grouped = df.reset_index().groupby("Code", as_index=False)
    fixed = grouped.apply(apply_func).reset_index()
    fixed.drop(["level_0", "level_1"], axis=1, inplace=True)
    return fixed 
    
def remove_scars(df):
    safe_countries = set(df["Code"].unique()) - set(scar_ranges.keys())
    valid = df[df["Code"].isin(safe_countries)].copy()
    valid.drop(["Country", "Code"], axis=1, inplace=True)
    return valid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to data file", default="joined_data.csv")
    args = parser.parse_args()
    
    df = pd.read_csv(args.data, index_col=0)
    print(df)
    df = fill_na_simple(df)
    
    fwd_frames = {}

    fwd_fixed = fix_df(df, add_previous_years)
    for country in scar_ranges:
        country_df = fwd_fixed[fwd_fixed["Code"] == country].copy()
        country_df.drop(["Country", "Code"], axis=1, inplace=True)
        fwd_frames[country] = country_df
    
    forward_valid = remove_scars(fwd_fixed)
    fwd_results = {}
    for code in fwd_frames:
        country_frame = fwd_frames[code]
        results = forward_predict(forward_valid, country_frame)
        fwd_results[code] = results
    
    # Dump the results so that we can graph
    dict_bytes = pickle.dumps(fwd_results)
    with open("fwd_results.pickle", "wb") as f:
        f.write(dict_bytes)
    
    back_frames = {}
    back_fixed = fix_df(df, add_next_years)
    for country in scar_ranges:
        country_df = back_fixed[back_fixed["Code"] == country].copy()
        country_df.drop(["Country", "Code"], axis=1, inplace=True)
        back_frames[country] = country_df

    back_valid = remove_scars(back_fixed)
    back_results = {}
    for code in back_frames:
        country_frame = back_frames[code]
        results = back_predict(back_valid, country_frame)
        back_results[code] = results

    # Dump the results so that we can graph
    dict_bytes = pickle.dumps(back_results)
    with open("back_results.pickle", "wb") as f:
        f.write(dict_bytes)