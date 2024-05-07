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

Prediction_Forecast generates 3 prediction models: Decision Tree Regressor, Random Forest Regressor, and Gradient Boosted Regressor. It then backtests the results, and compares the result to the actual predictions.

Two main pieces of data:
country_record: map from countries (located in notebook) to a 2D array. The first dimension is the type of classifier (Decision Tree Regressor, Random Forest Regressor, and Gradient Boosted Regressor), and the second dimension is backprediction from 2020 to 1960.
country_pop: map from countries to a 1D array, which records the actual population figures of the country