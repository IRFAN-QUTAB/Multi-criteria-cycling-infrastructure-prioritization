"""
export_road_lts_composition.py - Per-road LTS composition for the
mechanism analysis (finding 2: structural independence).

For every named road, exports:
  - total length (m) and segment count
  - length that is driveable (car network)
  - length that is bike-allowed
  - length that is low-stress (bike_allowed AND lts <= 2)
  - dual PageRank (centrality rank of the road)

Usage (run once per city, with that city's DB active):
  python export_road_lts_composition.py -n bolt://localhost:7687 \
      -u neo4j -p PASSWORD --city modena
Output: <city>_road_lts_composition.csv in the current folder.
"""
import argparse
import csv
from neo4j import GraphDatabase

Q = """
MATCH (a:RoadJunction)-[r:ROUTE]->(b:RoadJunction)
WHERE r.name IS NOT NULL AND r.status = 'active'
WITH r.name AS road_name,
     count(r) AS segments,
     sum(coalesce(r.distance, 0)) AS total_m,
     sum(CASE WHEN r.driveable = true
              THEN coalesce(r.distance, 0) ELSE 0 END) AS car_m,
     sum(CASE WHEN r.bike_allowed = true
              THEN coalesce(r.distance, 0) ELSE 0 END) AS bike_m,
     sum(CASE WHEN r.bike_allowed = true AND r.lts IS NOT NULL
              AND r.lts <= 2
              THEN coalesce(r.distance, 0) ELSE 0 END) AS bike_safe_m
OPTIONAL MATCH (ro:RoadOsm {name: road_name})
RETURN road_name, segments,
       round(total_m, 1) AS total_m,
       round(car_m, 1) AS car_m,
       round(bike_m, 1) AS bike_m,
       round(bike_safe_m, 1) AS bike_safe_m,
       ro.dual_pagerank AS dual_pagerank
ORDER BY road_name
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neo4jURL", "-n", required=True)
    ap.add_argument("--neo4juser", "-u", required=True)
    ap.add_argument("--neo4jpwd", "-p", required=True)
    ap.add_argument("--city", "-c", required=True)
    args = ap.parse_args()

    driver = GraphDatabase.driver(
        args.neo4jURL, auth=(args.neo4juser, args.neo4jpwd),
        notifications_min_severity="OFF")
    with driver.session() as s:
        rows = [dict(r) for r in s.run(Q)]
    driver.close()

    out = "{}_road_lts_composition.csv".format(args.city.lower().strip())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Exported {} roads -> {}".format(len(rows), out))

    # Quick headline number: share of the low-stress network that lies
    # on the top-50% most central roads (the ones we disrupt).
    ranked = [r for r in rows if r["dual_pagerank"] is not None]
    ranked.sort(key=lambda r: r["dual_pagerank"], reverse=True)
    top = ranked[: max(1, len(ranked) // 2)]
    total_safe = sum(r["bike_safe_m"] or 0 for r in rows)
    top_safe = sum(r["bike_safe_m"] or 0 for r in top)
    total_car = sum(r["car_m"] or 0 for r in rows)
    top_car = sum(r["car_m"] or 0 for r in top)
    if total_safe > 0 and total_car > 0:
        print("Low-stress km on top-50% central roads: {:.1f}%".format(
            100.0 * top_safe / total_safe))
        print("Car km on top-50% central roads:        {:.1f}%".format(
            100.0 * top_car / total_car))


if __name__ == "__main__":
    main()
