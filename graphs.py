import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import tikzplotlib
from scipy.interpolate import BSpline, make_interp_spline
from sklearn.metrics import mean_squared_error, r2_score

plt.style.use("./paper.mplstyle")

scar_ranges = {
    "W140000WOKH": (1972, 1985),
    "W140000WORU": (1995, 2015),
    "W140000WORW": (1992, 2005),
}

titles = {
    "W140000WOKH": "Cambodia",
    "W140000WORU": "Russia",
    "W140000WORW": "Rwanda",
}

colors = {
    "Actual": "#DA3E52",
    "Decision tree": "#7E6B8F",
    "Gradient boost": "#96E6B3",
    "Random forest": "#A3D9FF",
}

def generate_zoomed_plots(data, title):
    fig, ax = plt.subplots(nrows=1, ncols=len(data), layout="constrained", dpi=300)
    for i, country in enumerate(data):
        _ax = ax[i]
        generate_country_lines(country, data[country], _ax, zoom=(scar_ranges[country][0], scar_ranges[country][1]))
        
    handles, labels = ax[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, ncol=2, loc='outside lower center')
    leg.get_frame().set_linewidth(0.0)
    save_title = title.replace(" ", "_").lower()
    plt.savefig(f"{save_title}_zoomed.png")
    tikzplotlib.save(f"{save_title}_zoomed.tex")

def generate_country_lines(country, data, ax, zoom=None):
    ax.set_title(country)
    print(data)
    models = set(data.keys()) - set(["Index", "Actual"])
    indices = np.array(data["Index"][1:])
    actual = data["Actual"][1:]
    xs = [("Actual", actual)]
    r2_scores = {}
    for model in models:
        xs.append((model, data[model][1:])) 
        r2 = r2_score(actual, data[model][1:])
        r2_scores[model] = r2
        
    if zoom:
        start_idx, end_idx = indices.searchsorted(zoom[0]), indices.searchsorted(zoom[1])
        print(start_idx, end_idx)
        indices = indices[start_idx:end_idx]
        for i, (label, x) in enumerate(xs):
            xs[i] = (label, x[start_idx:end_idx])

    for label, x in xs:
        xnew = np.linspace(indices.min(), indices.max(), 1000) 
        spl = make_interp_spline(indices, x, k=3)  # type: BSpline
        smoothed = spl(xnew)
        ax.plot(xnew, smoothed, label=label, linewidth=1.5, color=colors[label])

    # ax.legend()
    ax.set_xlabel("Year")
    ax.set_ylabel("Population")
    ax.set_title(titles[country])
    
    return r2_scores

def normalize(pop):
    return (pop - 2705) / (1413142846 - 2705)

def generate_lines(data, title):
    fig, ax = plt.subplots(nrows=1, ncols=len(data), layout="constrained", dpi=300)
    r2 = {country: None for country in data}
    for i, country in enumerate(data):
        _ax = ax[i]
        country_r2 = generate_country_lines(country, data[country], _ax)
        r2[country] = country_r2

    for country in r2:
        print(f"{country}:")
        for model in r2[country]:
            print(f"{model}:", r2[country][model])

    handles, labels = ax[0].get_legend_handles_labels()
     
    leg = fig.legend(handles, labels, ncols=2, loc='outside lower center')
    leg.get_frame().set_linewidth(0.0)

    save_title = title.replace(" ", "_").lower()
    plt.savefig(f"{save_title}.png")
    tikzplotlib.save(f"{save_title}.tex")
         
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="path to data file", type=str)
    parser.add_argument("--title", help="title of the graph", type=str)
    args = parser.parse_args()
    
    data = {}
    with open(args.data, "rb") as f:
        data = pickle.load(f)
        
    generate_zoomed_plots(data, args.title)
    generate_lines(data, args.title)