# A Simulation-Driven Multi-Criteria Framework for Prioritizing Low-Stress Cycling Infrastructure under Road Closures

A Python + Neo4j framework in which a disruption simulation quantifies the criteria of a multi-criteria decision-making (MCDM) model. It is applied to rank the roads of a city for parallel low-stress cycling routes.

The framework builds a multiplex urban road graph from OpenStreetMap. In this graph, the car and cycling layers share the same street edges, and every cycling edge is graded by Level of Traffic Stress (LTS). Each candidate road is closed one at a time and in groups. The simulation measures accessibility to essential services for all residential origins, weighted by census population, on three layers: car, cycling on all permitted roads, and cycling on low-stress roads only. Five criteria derived from the simulation are weighted with CRITIC and aggregated by NR-TOPSIS, VIKOR, and PROMETHEE II into a consensus ranking. The ranking is tested with sensitivity, comparative and rank-reversal analyses, out-of-sample simulated closures, and two intervention experiments (parallel bypass and road upgrade). The framework is applied to three Italian cities: Modena, Ferrara, and Pesaro.

## Requirements

- [Neo4j Desktop](https://neo4j.com/docs/operations-manual/current/installation/) (DBMS version 5.26+)
- [Graph Data Science](https://neo4j.com/docs/graph-data-science/current/installation/) plugin (version 2.13+)
- [APOC](https://neo4j.com/labs/apoc/) plugin
- Python 3.8+

### Neo4j Configuration

Edit your DBMS settings and add the following lines after the Other Neo4j system properties section:

```
apoc.import.file.enabled=true
dbms.security.procedures.unrestricted=jwt.security.*,apoc.*,gds.*,spatial.*
```

### Python Dependencies

```shell
pip install osmnx overpy neo4j pandas numpy scipy matplotlib seaborn pyshp openpyxl pyproj pymcdm
```

---

## Pipeline Overview

```
1. Create Road Network Graph     →  Multimodal graph (car + bike + LTS) in Neo4j
2. Add POIs to Graph             →  Essential services linked to the network
3. Create Dual Graph             →  RoadOsm nodes + CONNECTED edges
4. Compute Centrality            →  PageRank, Betweenness, Degree on the dual graph
5. Export the graph              →  One read-only export per city (last Neo4j step)
6. Disruption simulation         →  Single-road closures (9 set-ups) + simultaneous closures
7. MCDM analysis                 →  Criteria, weights, rankings, sensitivity, rank reversal, SMAA
8. Simulation checks/validation  →  Network statistics, asymmetry, held-out closures, bypass, upgrade
9. Figures                       →  All figures of the paper
```

Optional exports for mapping: Export Road LTS Composition, Export for QGIS (see the end of this file).

Folder layout used by Steps 5–9:

```
ESWA/
├── mcdm_analysis.py          # MCDM methods (imported by mcdm_v2.py)
├── verify_mcdm_methods.py    # check of the methods against pymcdm
└── rerun/
    ├── export_neo4j_city.py
    ├── run_city.py
    ├── mmsim/                # simulation engine (graph, LTS, OD pairs, closures, routing)
    ├── analysis/             # metrics.py, mcdm_v2.py, sim_checks.py, plot_v2.py
    └── outputs_v2/<city>/    # results used in the paper
```

The census population files `modena_population.csv`, `ferrara_population.csv`, and `pesaro_population.csv` (columns `lat, lon, population`, census-section centroids from the ISTAT 2021 census) must be in the project root.

---

## 1. Create Road Network Graph

Downloads the OSM road network for a given center point and radius. It classifies every edge for car driveability, bicycle access, and Level of Traffic Stress (LTS 1–4) following Vierø & Szell (2025), and writes the graph into Neo4j. Shared-lane markings (sharrows) are treated as mixed traffic, and a bus route raises the LTS by one level.

```shell
python "Create Graph.py" -x 44.645885 -y 10.9255707 -d 5 \
    -n bolt://localhost:7687 -u neo4j -p <password> -f modena
```

| Parameter | Description |
|-----------|-------------|
| `-x` | Latitude of the center point |
| `-y` | Longitude of the center point |
| `-d` | Radius in km |
| `-n` | Neo4j bolt URL |
| `-u` | Neo4j username |
| `-p` | Neo4j password |
| `-f` | City name / file identifier |

**Edge properties set:** `highway`, `distance`, `driveable`, `bike_allowed`, `lts` (1–4), `name`, `oneway`, `surface`, `lanes`, `maxspeed`, `bridge`, `has_bus_route`, `cycleway`, `status='active'`.

---

## 2. Add POIs to Graph

Queries essential services (hospitals, clinics, doctors, pharmacies, schools, kindergartens, universities, police and fire stations, post offices, libraries, banks, supermarkets, convenience stores) from the Overpass API and links them to the road network.

```shell
python "Import POIs to the network.py" -n bolt://localhost:7687 -u neo4j -p <password> \
    -x 44.645885 -y 10.9255707 -d 5
```

---

## 3. Create Dual Graph

Transforms the junction graph into a dual (named-road) graph. Each named road becomes a `RoadOsm` node, and two roads that share a junction are connected by a `CONNECTED` edge. Connections through roundabouts and unnamed segments are handled.

```shell
python "Creation of Dual Graph.py" -n bolt://localhost:7687 -u neo4j -p <password>
```

---

## 4. Compute Centrality

Runs centrality metrics on the dual graph with Neo4j GDS and writes them back as node properties: `Pagerank`, `Betweenness`, `Degree` on `RoadOsm` nodes, and junction betweenness `bc` on `RoadJunction` nodes, which is used to choose the closed segments.

```shell
python "Compute Centrality.py" -n bolt://localhost:7687 -u neo4j -p <password>
```

---

## 5. Export the Graph (last step that needs Neo4j)

Reads the complete graph of a city (nodes, relationships, all properties) and saves it to `ESWA/rerun/graph_exports/<city>/`. The database is only read, never modified. Start the database of the city in Neo4j Desktop first.

```shell
python ESWA/rerun/export_neo4j_city.py --city modena -n bolt://localhost:7687 -u neo4j -p <password>
```

`run_city.py` (Step 6) runs this export automatically if it has not been done yet.

---

## 6. Disruption Simulation

```shell
python ESWA/rerun/run_city.py --city modena -p <password>
```

The password is needed only if the export of Step 5 does not exist yet. The run is resumable: if it stops, the same command continues from where it stopped. Progress is written to `ESWA/rerun/results_v2/<city>/log.txt`.

| Parameter | Description |
|-----------|-------------|
| `--city` | `modena`, `ferrara`, or `pesaro` |
| `-n`, `-u`, `-p` | Neo4j URL, user, password (export only) |
| `--database` | Named Neo4j database, if not the default |
| `--workers` | Number of parallel processes |
| `--stages` | Stages to run (default: `export,prepare,check,single,asym`) |
| `--smoke N` | Quick test with N scenarios per stage |

| Stage | What it does |
|-------|--------------|
| `export` | Graph export (Step 5) |
| `prepare` | OD pairs, service access, alternatives |
| `check` | Reproduces the baseline of an earlier Neo4j-based run, if its files are present; otherwise skipped |
| `single` | Closes every alternative road, in 9 set-ups |
| `asym` | Simultaneous closures of the top-k roads by PageRank and of random roads |

**Simulation design:**

- **OD pairs.** Every residential junction is an origin and receives a population weight: the residents of each census section are divided equally among the residential junctions for which that section is the nearest. Each origin is paired with the nearest essential service by car network distance, chosen among its three nearest services by straight-line distance. The pairs are split at random into set A (criteria) and set B (held out for validation). This gives 8,137 pairs in Modena, 8,422 in Ferrara, and 5,955 in Pesaro.
- **Alternatives.** The top half of the named roads by dual PageRank: 769 in Modena, 594 in Ferrara, 534 in Pesaro.
- **Closures.** A closure removes physical street segments in both directions. Roads with at most ten segments are closed entirely. For longer roads, 25% of the segments (at least ten) next to the junctions with the highest betweenness are closed. Trips whose origin or destination junction is isolated by the closure are excluded.
- **Routing.** Shortest paths by distance with `scipy.sparse.csgraph` Dijkstra. After a closure, only the pairs whose intact path uses a closed edge are re-routed, which is exact.
- **Simultaneous closures.** k = 5, 10, 20, 30, 50, 70, 100, against 20 random selections per k from all named roads and from the alternatives.
- **Set-up variants (single closures).** `main` (all pairs); closure depth 10%, 50%, 100%; stress-weighted cycling costs (`cost_lts`); LTS without the sharrow rule (`lts_stored`); 30 km/h default on residential streets (`lts_zone30`); same access junction for all layers (`access_legacy`); destination chosen in each layer (`dest_layer`).

---

## 7. MCDM Analysis

First verify the MCDM implementations against `pymcdm` (must end with `ALL CHECKS PASSED`):

```shell
python ESWA/verify_mcdm_methods.py
```

Then run the analysis:

```shell
python ESWA/rerun/analysis/mcdm_v2.py
```

| Parameter | Description |
|-----------|-------------|
| `--cities` | Cities to analyze (default: all three) |
| `--primary` | Primary weighting scheme (default: `critic`) |
| `--smaa-samples` | Number of SMAA-2 weight samples (default: 2000) |
| `--rr-iter` | Random-deletion repetitions for the rank-reversal test (default: 200) |

### Decision model

| Element | Definition |
|---------|------------|
| Alternatives | Roads in the top half by dual PageRank |
| C1 Network importance | Dual PageRank of the road |
| C2 Detour | Population-weighted mean excess detour of the low-stress trips that remain reachable (set A) |
| C3 Service loss | Population-weighted share of low-stress trips that lose access to their service (set A) |
| C4 Fragmentation | Population share cut off from the main low-stress component, excluding junctions isolated by the closure |
| C5 Compensation failure | max(0, H_safe − H_car), with H = adjusted detour ratio − 1 (lost trips counted with λ = 5) |
| Weights | CRITIC (primary); entropy, CRITIC–entropy, and equal weights for sensitivity |
| Core methods | NR-TOPSIS, VIKOR (v = 0.5), PROMETHEE II (Gaussian preference); Borda-type consensus, ties broken by NR-TOPSIS |
| Comparison methods | TOPSIS, WSM, EDAS, MABAC, MARCOS |
| Benchmarks | Dual PageRank, car-only MCDM, heuristic investment priority score (IPS) |

All criteria are of benefit type. The NR-TOPSIS bounds are fixed on the candidate set: [0, max] for C2–C5 and the observed range for C1.

**Analyses:**

- Criteria weights; rankings of all eight methods and the consensus.
- Rank similarity (Spearman, Kendall, WS coefficient, Kendall's W) and weighting-method comparison.
- 30 weight scenarios, one-at-a-time weight changes, and criteria-group scenarios with criteria contribution.
- Rank reversal, with fixed and with re-estimated weights and bounds, and under random deletion.
- Method parameters and SMAA-2.
- Benchmark comparison.
- Robustness to the simulation set-up, the model choices (C5 variants, cycling layers, exclusion rule), the OD pairs (set B, five subsets of set A), and boundary roads.

**Output:** `ESWA/rerun/outputs_v2/<city>/` (including `scores_rankings.csv` and `rankings_for_validation.csv`) and cross-city summaries `ESWA/rerun/outputs_v2/summary_*.csv`.

---

## 8. Simulation Checks and Validation

Run after Step 7, because the validation uses the rankings produced there.

```shell
python ESWA/rerun/analysis/sim_checks.py --city modena
```

| Parameter | Description |
|-----------|-------------|
| `--city` | City to analyze |
| `--what` | Analyses to run (default: `stats,asym,validate,invest,bypass`) |

| Analysis | What it does |
|----------|--------------|
| `stats` | Network statistics, coverage, connectivity, and route overlap of the layers |
| `asym` | Targeted vs random simultaneous closures, with penalty λ = 2–20 |
| `validate` | Closes the top-k roads (k = 5–50) of each ranking and of 20 random selections; accessibility measured on the held-out set B; paired bootstrap confidence intervals |
| `bypass` | Adds a parallel LTS 1 route (1.3 × segment length; 1.1 and 1.5 as sensitivity) along each top-20 road and 200 random alternatives, closes the road, and measures the avoided low-stress harm on set B |
| `invest` | Upgrades the top-k roads of each ranking to LTS 1 and measures the share of trips that gain low-stress access |

---

## 9. Figures

```shell
python ESWA/rerun/analysis/plot_v2.py
```

Writes all figures of the paper (PNG and PDF) to `ESWA/rerun/figures_v2/`.

---

## Optional Exports

**Road LTS composition** (per-road length by layer and PageRank):

```shell
python Export_Road_LTS_Composition.py -n bolt://localhost:7687 -u neo4j -p <password> --city modena
```

**QGIS** (georeferenced CSVs for maps):

```shell
python Export_QGIS.py -n bolt://localhost:7687 -u neo4j -p <password> \
    --city modena --output qgis_export --baseline --lts
```

---

## Example: Three Italian Cities

```shell
# Build the graphs (Steps 1–4), once per city
python "Create Graph.py" -x 44.645885 -y 10.9255707 -d 5 -n bolt://localhost:7687 -u neo4j -p <password> -f modena
python "Create Graph.py" -x 44.835888 -y 11.619787  -d 5 -n bolt://localhost:7687 -u neo4j -p <password> -f ferrara
python "Create Graph.py" -x 43.910156 -y 12.913389  -d 5 -n bolt://localhost:7687 -u neo4j -p <password> -f pesaro

# Simulation (start the database of each city before its run)
for city in modena ferrara pesaro; do
    python ESWA/rerun/run_city.py --city $city -p <password>
done

# MCDM, validation, figures (offline)
python ESWA/verify_mcdm_methods.py
python ESWA/rerun/analysis/mcdm_v2.py
for city in modena ferrara pesaro; do
    python ESWA/rerun/analysis/sim_checks.py --city $city
done
python ESWA/rerun/analysis/plot_v2.py
```
