# A Simulation-Driven Multi-Criteria Framework for Prioritizing Low-Stress Cycling Infrastructure under Road Closures

A decision-support framework in which a disruption simulation quantifies the criteria of a multi-criteria decision-making (MCDM) model. It ranks the roads of a city for parallel low-stress cycling routes.

A multiplex road graph is built from OpenStreetMap: the car and cycling layers share the same street edges, and every cycling edge is graded by Level of Traffic Stress (LTS). Each candidate road is closed, and the loss of accessibility to essential services is measured for all residential origins, weighted by census population. Five criteria derived from the simulation are weighted with CRITIC and aggregated by NR-TOPSIS, VIKOR, and PROMETHEE II into a consensus ranking. The framework is applied to Modena, Ferrara, and Pesaro (Italy).

## Requirements

- Neo4j 5.26+ with the Graph Data Science (2.13+) and APOC plugins
- Python 3.8+

```shell
pip install -r requirements.txt
```

Neo4j is used to build the graph. All later steps run offline on an export of the graph.

## Repository structure

```
Create Graph.py                  multimodal graph with LTS grades
Import POIs to the network.py    essential services
Creation of Dual Graph.py        named-road (dual) graph
Compute Centrality.py            PageRank and betweenness
Export_Road_LTS_Composition.py   optional: per-road LTS composition
Export_QGIS.py                   optional: CSVs for maps in QGIS
export_neo4j_city.py             export of the graph from Neo4j
run_city.py                      disruption simulation
mmsim/                           simulation engine
analysis/                        MCDM model, validation, figures
data/                            census population per city (see data/README.md)
```

## Pipeline

**1. Build the graph in Neo4j**

```shell
python "Create Graph.py" -x 44.645885 -y 10.9255707 -d 5 -n bolt://localhost:7687 -u neo4j -p <password> -f modena
python "Import POIs to the network.py" -x 44.645885 -y 10.9255707 -d 5 -n bolt://localhost:7687 -u neo4j -p <password>
python "Creation of Dual Graph.py" -n bolt://localhost:7687 -u neo4j -p <password>
python "Compute Centrality.py" -n bolt://localhost:7687 -u neo4j -p <password>
```

City centers (radius 5 km): Modena `44.645885, 10.9255707`; Ferrara `44.835888, 11.619787`; Pesaro `43.910156, 12.913389`.

**2. Disruption simulation**

```shell
python run_city.py --city modena -p <password>
```

The graph is exported once (`export_neo4j_city.py`), and the simulation runs on the export:

- **OD pairs:** all residential origins with population weights, each paired with its nearest essential service, split into set A (criteria) and set B (validation).
- **Alternatives:** the top half of the named roads by dual PageRank.
- **Single-road closures:** 25% of the segments of each road, plus 8 set-up variants (closure depth, stress-weighted cost, LTS rules, speed defaults, access and destination rules).
- **Simultaneous closures:** the top-k roads by PageRank and random roads, k = 5–100.

**3. MCDM analysis and validation**

```shell
python analysis/sim_checks.py --city modena --what stats
python analysis/mcdm.py
python analysis/sim_checks.py --city modena
python analysis/verify_mcdm_methods.py
python analysis/plot_figures.py
```

| Element | Definition |
|---------|------------|
| C1 Network importance | Dual PageRank of the road |
| C2 Detour | Excess detour of the low-stress trips that remain reachable |
| C3 Service loss | Share of low-stress trips that lose access to their service |
| C4 Fragmentation | Population share cut off from the main low-stress component |
| C5 Compensation failure | max(0, H<sub>safe</sub> − H<sub>car</sub>), H = adjusted detour ratio − 1 |
| Weights | CRITIC (entropy, CRITIC–entropy, equal for sensitivity) |
| Core methods | NR-TOPSIS, VIKOR, PROMETHEE II; Borda-type consensus |
| Comparison methods | TOPSIS, WSM, EDAS, MABAC, MARCOS |
| Benchmarks | Dual PageRank, car-only MCDM, heuristic priority score |

- `mcdm.py`: weights, rankings, rank similarity, weight scenarios, criteria contribution, rank reversal, SMAA-2, and robustness to the simulation set-up and the OD pairs.
- `sim_checks.py`: network statistics, targeted vs random closures, and the validation:
  - closures of the top-ranked roads on the held-out trips;
  - a parallel low-stress bypass along the top-ranked roads;
  - an upgrade of the top-ranked roads to low stress.
- `verify_mcdm_methods.py`: checks the MCDM implementations against `pymcdm`.
- `plot_figures.py`: the figures of the paper.

Results are written to `results/`, `outputs/`, and `figures/`.

## Data

OpenStreetMap data are available under the Open Database License. Census data are published by the Italian National Institute of Statistics (ISTAT).
