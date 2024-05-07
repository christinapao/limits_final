# limits_final

Please reach out if there are any questions or concerns with the code in this project. We used Python for all steps of the project; the packages we used can be found in `requirements.txt`.

## Data

We used two sources for our data:

- The IDB, found [here](https://www.census.gov/programs-surveys/international-programs/about/idb.html). The provided link downloads two files; we used `idb5yr.txt` for all our analysis.
- The World Bank World Development Indicators, found [here](https://databank.worldbank.org/source/world-development-indicators). We used a subset of the possible indicators.
- The result of joining the two datasets is in `joined_data.csv`.
- We also provide the file `codes.csv`, which we used as part of the join process
- To reproduce the above file, two steps need to be taken:
    1. Clean and manipulate the world bank data: `python analysis.py -d PATH_TO_WORLD_BANK_DATA`
    2. Join the two datasets together: `python join.py --input PATH_TO_OUTPUT_STEP_1 --idb PATH_TO_IDB_5YR_FILE`

## Models

- The file `Forecast_population.ipynb` is a Jupyter Notebook used during the model development process. The same functionality is produced as a script in `model.py`. The two are functionally identical. For simplicity's sake, we discuss `model.py` here.
- The script should run without any arguments under the assumption that the joined data being used is `joined_data.csv`. If not, it takes an optional parameter to specify an input file:
    * `python model.py --data PATH_TO_DATA`
- The script performs the training and prediction for all six models used in our experiments: Gradient boost, random forest, and decision tree for both forward and backwards prediction. Results are outputted as pickle files which can be used in the next step to reproduce the graphs included in our writeup.

## Graphs

NOTE: We had to manually tweak the tikzplotlib package due to its reliance on a deprecated function in the current version of matplotlib (a three line change). This is not included in the repository, and the graphing script should function without it as long as the tikzplotlib specific code is omitted. Should this prove problematic, please contact us!

- Producing the graphs should be simple: `graph.py` takes as input a data file and a title and produces both the centered and full graphs for the given data file.
    * To produce the forward graphs: `python graph.py --data PATH_TO_FORWARD_RESULTS.pickle --title TITLE`. Output in `title.tex/png` and `title_zoomed.tex/png`
    * To produce the back graphs: `python graph.py --data PATH_TO_BACKWARD_RESULTS.pickle --title TITLE`. Output in `title.tex/png` and `title_zoomed.tex/png`
