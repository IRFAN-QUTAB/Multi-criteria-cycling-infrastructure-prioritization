# Coverage Is Not Resilience: Asymmetric Vulnerability of Multiplex Car–Bicycle Networks Under Targeted Road Disruptions

A Python + Neo4j framework for quantifying how car and bicycle networks respond differently to road closures. It constructs a multiplex urban road graph from OpenStreetMap with Level of Traffic Stress (LTS) classification, computes dual-graph centrality, simulates targeted road disruptions, and measures modal accessibility loss across origin–destination pairs to essential services. Applied to three Italian cities (Modena, Ferrara, Pesaro).

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
pip install osmnx overpy neo4j pandas numpy scipy matplotlib seaborn pyshp openpyxl pyproj
```

---

## Pipeline Overview

The framework follows a sequential pipeline. Each step updates the Neo4j database for the next step.

```
1. Create Road Network Graph       →  Multi-modal graph (car + bike + LTS) in Neo4j
2. Add POIs to Graph               →  NEAR_CAR / NEAR_BIKE proximity edges
3. Create Dual Graph               →  RoadOsm nodes + CONNECTED edges
4. Compute Centrality              →  PageRank, Betweenness, Degree on dual graph
5. Vulnerability Analysis          →  One-by-one + simultaneous + null model results
6. Export Road LTS Composition     →  Per-road LTS breakdown CSV
7. Export for QGIS                 →  Spatial CSVs for map visualization
```

---

## 1. Create Road Network Graph

Downloads the OSM road network for a given center point and radius, classifies every edge for car driveability, bicycle access, and Level of Traffic Stress (LTS 1–4) following Vierø & Szell (2025), and writes the graph directly into Neo4j. Bus route detection adjusts LTS for shared corridors.

```shell
python "Create Graph.py" -x 44.645885 -y 10.9255707 -d 5 \
    -n bolt://localhost:7687 -u neo4j -p password -f modena
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

Queries essential amenities (hospitals, schools, pharmacies, supermarkets, banks, etc.) from the Overpass API and links each POI to the nearest road junction via `NEAR_CAR` and `NEAR_BIKE` edges, enabling mode-specific OD pair resolution.

```shell
python "Import amenity.py" -n bolt://localhost:7687 -u neo4j -p password \
    -x 44.645885 -y 10.9255707 -d 5
```

| Parameter | Description |
|-----------|-------------|
| `-n` | Neo4j bolt URL |
| `-u` | Neo4j username |
| `-p` | Neo4j password |
| `-x`, `-y` | Center latitude and longitude |
| `-d` | Radius in km |

**Output:** POI nodes with `:Tag` labels and `NEAR_CAR` / `NEAR_BIKE` edges to the nearest driveable and bikeable junctions respectively.

---

## 3. Create Dual Graph

Transforms the primal junction graph into a dual (named-road) graph where each named road becomes a `RoadOsm` node and two roads sharing a junction are connected by a `CONNECTED` edge. Handles connections through roundabouts and unnamed segments.

```shell
python "Dual Graph (Named-Road).py" -n bolt://localhost:7687 -u neo4j -p password
```

| Parameter | Description |
|-----------|-------------|
| `-n` | Neo4j bolt URL |
| `-u` | Neo4j username |
| `-p` | Neo4j password |

**Output:** `RoadOsm` nodes with segment counts and total length, `CONNECTED` edges with weight.

---

## 4. Compute Centrality

Runs graph-theoretic centrality metrics on the dual graph using Neo4j GDS projections and writes the results back as node properties.

```shell
python compute_centrality.py -n bolt://localhost:7687 -u neo4j -p password
```

| Parameter | Description |
|-----------|-------------|
| `-n` | Neo4j bolt URL |
| `-u` | Neo4j username |
| `-p` | Neo4j password |
| `--top` | Number of top roads to display (default: 20) |

**Computed metrics:** `Pagerank`, `Betweenness`, `Degree` on `RoadOsm` nodes. Junction-level `bc` (betweenness centrality) on `RoadJunction` nodes for percentage-based closure targeting.

---

## 5. Vulnerability Analysis

The core analysis. Selects residential-to-POI origin–destination pairs, ranks roads by PageRank, and simulates targeted road closures measuring accessibility loss across three routing modes: car (`driveable=true`), bike safe (`bike_allowed=true AND lts≤2`), and bike all (`bike_allowed=true`).

```shell
python vulnerability_analysis.py -n bolt://localhost:7687 -u neo4j -p password \
    --city modena --od-pairs 200 --top-pct 50 \
    --closure-pct 0.25 --closure-min 10 \
    --weight distance --unreachable-penalty 5.0 \
    --null-iterations 5 --seed 42
```

| Parameter | Description |
|-----------|-------------|
| `--city` | City name (used in output filenames) |
| `--od-pairs` | Number of OD pairs to sample (default: 200) |
| `--top-pct` | Top X% of roads by PageRank to analyze (default: 50) |
| `--closure-pct` | Fraction of a road's segments to close (default: 0.25) |
| `--closure-min` | Minimum segments to close per road (default: 10) |
| `--weight` | Edge impedance: `distance`, `time`, or `lts` (default: distance) |
| `--unreachable-penalty` | Detour ratio for newly-unreachable OD pairs (default: 5.0) |
| `--population-csv` | ISTAT census CSV for population-weighted origins |
| `--poi-candidates` | Nearest-POI candidates resolved by network distance (default: 3) |
| `--null-iterations` | Random null-model iterations per k, 0 disables (default: 5) |
| `--seed` | Random seed for reproducibility (default: 42) |
| `--sensitivity` | 10%, 25%, 50% closure and save comparison |

**Analysis components:**

- **One-by-one closure:** Each top road closed individually (25% of segments by junction betweenness). Measures detour ratio, POIs lost, WCC fragmentation, path overlap, compensation category, MID, and investment priority score.
- **Simultaneous closure:** Top k roads closed together (k = 10, 20, 30, 50, 70, 100). Percolation-style analysis.
- **Null model:** k random roads closed (5 iterations) for targeted vs. random comparison.
- **Per-OD vulnerability:** Tracks which OD pairs are most frequently affected across all closures.

**Key metrics:**

| Metric | Description |
|--------|-------------|
| BIAI | Bike Infrastructure Adequacy Index = bike_safe reachable / car reachable at baseline |
| MID | Modal Impact Differential = bike detour − car detour per road closure |
| Compensation | full / partial / none / bike_only_vulnerable / no_impact |
| Investment Priority Score | `pagerank × max(car_impact, bike_impact) × (0.5 + bike_fail) + wcc_penalty` |

**Output per city:**

| File | Contents |
|------|----------|
| `<city>_<ts>_pct25.json` | Full results with baseline, BIAI, WCC, pooled Wilcoxon |
| `<city>_<ts>_pct25.csv` | Per-road summary sorted by investment priority |
| `<city>_<ts>_pct25_simultaneous.json/csv` | Simultaneous closure results |
| `<city>_<ts>_pct25_null_model.json/csv` | Random null model (mean ± s.d.) |
| `<city>_<ts>_pct25_od_pairs.json` | Selected OD pairs with coordinates |
| `<city>_<ts>_pct25_od_vulnerability.csv` | Per-OD vulnerability ranking |

---

## 6. Export Road LTS Composition

Exports per-road LTS breakdown for mechanism analysis: total length, car/bike/safe-bike kilometres, and dual PageRank for each named road.

```shell
python export_road_lts_composition.py -n bolt://localhost:7687 -u neo4j -p password \
    --city modena
```

| Parameter | Description |
|-----------|-------------|
| `-n` | Neo4j bolt URL |
| `-u` | Neo4j username |
| `-p` | Neo4j password |
| `--city` | City name |

**Output:** `<city>_road_lts_composition.csv` with columns `road_name, segments, total_m, car_m, bike_m, bike_safe_m, dual_pagerank`.

---

## 7. Export for QGIS

Exports road-level and OD-level data as georeferenced CSVs for spatial visualization in QGIS.

```shell
python export_qgis.py -n bolt://localhost:7687 -u neo4j -p password \
    --city modena --output qgis_export --baseline --lts
```

| Parameter | Description |
|-----------|-------------|
| `--city` | City name |
| `--output` | Output folder (default: qgis_export) |
| `--baseline` | Export baseline edge data with LTS and driveability |
| `--lts` | Include LTS composition |
| `--dual` | Export dual graph nodes |
| `--pois` | Export POI locations |
| `--scenario` | Export a specific disruption scenario |


---

## Example: Three Italian Cities

**Modena** (5 km radius, ~95k edges):
```shell
python "Create Graph.py" -x 44.645885 -y 10.9255707 -d 5 \
    -n bolt://localhost:7687 -u neo4j -p password -f modena
```

**Ferrara** (5 km radius, ~83k edges):
```shell
python "Create Graph.py" -x 44.835888 -y 11.619787 -d 5 \
    -n bolt://localhost:7687 -u neo4j -p password -f ferrara
```

**Pesaro** (5 km radius, ~64k edges):
```shell
python "Create Graph.py" -x 43.910156 -y 12.913389 -d 5 \
    -n bolt://localhost:7687 -u neo4j -p password -f pesaro
```

Full vulnerability analysis for all three:
```shell
for city in modena ferrara pesaro; do
    python vulnerability_analysis.py -n bolt://localhost:7687 -u neo4j -p password \
        --city $city --od-pairs 200 --top-pct 50 --closure-pct 0.25
done
```
