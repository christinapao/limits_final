# limits_final

- `joined_data/` has the data that results from joining the world bank data with the international database. Currently just using variables from the `SP.POP` series in the world bank data, but can be easily changed
- `join.py` produces the files above. Usage: `python join.py --input PATH_TO_WORLDBANK_FILE --idb PATH_TO_IDB_5YR_FILE --code ISO2_COUNTRY_CODE`
    * Example: `python join.py --input cambodia.csv --idb idb5yr.txt --code kh`
- `analysis.py` produces the country-by-country csvs from the world bank data and turns them into the same general format as the idb data (rows as years, cols as variables)
