# Census population

`run_city.py` reads one file per city: `<city>_population.csv` (e.g. `modena_population.csv`), with the columns

```
lat,lon,population
```

One row per census section: the centroid of the section (WGS84) and its resident population. The files used in the paper were built from the 2021 Italian census sections (ISTAT, Basi territoriali 2021, field `POP21`).
