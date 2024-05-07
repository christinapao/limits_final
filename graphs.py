import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import tikzplotlib

def generate_country_lines(country, data, ax):
    ax.set_title(country)
    print(data)
    models = set(data.keys()) - set(["Index", "Actual"])
    indices = data["Index"][1:]
    actual = data["Actual"][1:]
    ax.plot(indices, actual, label="Actual")
    for model in models:
        ax.plot(indices, data[model][1:], label=model)
    ax.legend()

def generate_lines(data):
    fig, ax = plt.subplots(nrows=1, ncols=len(data), figsize=(8, 4))
    for i, country in enumerate(data):
        _ax = ax[i]
        generate_country_lines(country, data[country], _ax)
    plt.show()
         
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="path to data file", type=str)
    args = parser.parse_args()
    
    data = {}
    with open(args.data, "rb") as f:
        data = pickle.load(f)
        
    generate_lines(data)