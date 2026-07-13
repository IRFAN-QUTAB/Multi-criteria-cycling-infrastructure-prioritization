import argparse
import csv
import os
import time

from neo4j import GraphDatabase


class QGISExporter:
    """Export Neo4j graph edges as spatial CSVs for QGIS."""

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password),
            notifications_min_severity="OFF"
        )

    def close(self):
        self.driver.close()

    def _run(self, query, **params):
        with self.driver.session() as s:
            return list(s.run(query, **params))

    def export_primal_edges(self, output_path, include_disrupted=True):
        """Export all ROUTE edges with geometry as WKT LineString."""
        q = """
        MATCH (a:RoadJunction)-[r:ROUTE]->(b:RoadJunction)
        WHERE a.lat IS NOT NULL AND a.lon IS NOT NULL
          AND b.lat IS NOT NULL AND b.lon IS NOT NULL
        RETURN a.lon AS lon1, a.lat AS lat1,
               b.lon AS lon2, b.lat AS lat2,
               r.name AS name,
               r.highway AS highway,
               r.driveable AS driveable,
               r.bike_allowed AS bike_allowed,
               r.lts AS lts,
               r.distance AS distance,
               r.bridge AS bridge,
               CASE WHEN r.disrupted = true THEN true ELSE false END AS disrupted,
               r.disruption_type AS disruption_type
        """
        results = self._run(q)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "wkt", "name", "highway", "driveable", "bike_allowed",
                "lts", "distance", "bridge", "disrupted", "disruption_type"
            ])
            for r in results:
                if not include_disrupted and r["disrupted"]:
                    continue
                wkt = "LINESTRING({} {}, {} {})".format(
                    r["lon1"], r["lat1"], r["lon2"], r["lat2"])
                writer.writerow([
                    wkt,
                    r["name"] or "",
                    r["highway"] or "",
                    r["driveable"],
                    r["bike_allowed"],
                    r["lts"] if r["lts"] is not None else "",
                    round(r["distance"], 2) if r["distance"] else "",
                    r["bridge"] or "",
                    r["disrupted"],
                    r["disruption_type"] or "",
                ])

        print("  Exported {} edges to {}".format(len(results), output_path))

    def export_dual_connections(self, output_path):
        """Export dual graph CONNECTED edges with geometry (junction-to-junction)."""
        q = """
        MATCH (a:RoadOsm)-[c:CONNECTED]->(b:RoadOsm)
        WITH a, b, c
        // Get a representative junction for each road
        OPTIONAL MATCH (rj1:RoadJunction)-[r1:ROUTE {name: a.name}]->()
        WHERE rj1.lat IS NOT NULL
        WITH a, b, c, rj1 LIMIT 1
        OPTIONAL MATCH (rj2:RoadJunction)-[r2:ROUTE {name: b.name}]->()
        WHERE rj2.lat IS NOT NULL
        WITH a, b, c, rj1, rj2 LIMIT 1
        WHERE rj1 IS NOT NULL AND rj2 IS NOT NULL
        RETURN rj1.lon AS lon1, rj1.lat AS lat1,
               rj2.lon AS lon2, rj2.lat AS lat2,
               a.name AS road1, b.name AS road2,
               c.car_traversable AS car_trav,
               c.bike_traversable AS bike_trav,
               CASE WHEN a.disrupted = true OR b.disrupted = true
                    THEN true ELSE false END AS disrupted,
               a.disruption_type AS dtype1,
               b.disruption_type AS dtype2
        """
        results = self._run(q)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "wkt", "road1", "road2", "car_traversable",
                "bike_traversable", "disrupted", "disruption_type"
            ])
            for r in results:
                wkt = "LINESTRING({} {}, {} {})".format(
                    r["lon1"], r["lat1"], r["lon2"], r["lat2"])
                dtype = r["dtype1"] or r["dtype2"] or ""
                writer.writerow([
                    wkt, r["road1"], r["road2"],
                    r["car_trav"], r["bike_trav"],
                    r["disrupted"], dtype,
                ])

        print("  Exported {} dual connections to {}".format(len(results), output_path))

    def export_pois(self, output_path):
        """Export POIs with location for overlay on QGIS maps."""
        q = """
        MATCH (p:POI)-[:NEAR_CAR]->(rj:RoadJunction)
        WHERE rj.lat IS NOT NULL AND rj.lon IS NOT NULL
        OPTIONAL MATCH (p)-[:TAGS]->(t:Tag {key: 'amenity'})
        RETURN DISTINCT rj.lon AS lon, rj.lat AS lat,
               p.name AS name,
               t.value AS amenity_type
        """
        results = self._run(q)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["wkt", "name", "amenity_type"])
            for r in results:
                wkt = "POINT({} {})".format(r["lon"], r["lat"])
                writer.writerow([
                    wkt,
                    r["name"] or "",
                    r["amenity_type"] or "",
                ])

        print("  Exported {} POIs to {}".format(len(results), output_path))

    def export_lts_network(self, output_path, max_lts=2):
        """Export only the low-stress cycling network (LTS <= max_lts)."""
        q = """
        MATCH (a:RoadJunction)-[r:ROUTE]->(b:RoadJunction)
        WHERE r.bike_allowed = true AND r.lts IS NOT NULL AND r.lts <= $max_lts
          AND a.lat IS NOT NULL AND b.lat IS NOT NULL
        RETURN a.lon AS lon1, a.lat AS lat1,
               b.lon AS lon2, b.lat AS lat2,
               r.name AS name, r.lts AS lts,
               r.highway AS highway,
               CASE WHEN r.disrupted = true THEN true ELSE false END AS disrupted
        """
        results = self._run(q, max_lts=max_lts)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["wkt", "name", "lts", "highway", "disrupted"])
            for r in results:
                wkt = "LINESTRING({} {}, {} {})".format(
                    r["lon1"], r["lat1"], r["lon2"], r["lat2"])
                writer.writerow([
                    wkt, r["name"] or "", r["lts"],
                    r["highway"] or "", r["disrupted"],
                ])

        print("  Exported {} low-stress edges (LTS<={}) to {}".format(
            len(results), max_lts, output_path))


def main():
    parser = argparse.ArgumentParser(
        description="Export graph edges as spatial CSV for QGIS visualization"
    )
    parser.add_argument("--neo4jURL", "-n", required=True)
    parser.add_argument("--neo4juser", "-u", required=True)
    parser.add_argument("--neo4jpwd", "-p", required=True)
    parser.add_argument("--city", "-c", required=True)
    parser.add_argument("--output", "-o", default="qgis_export")
    parser.add_argument("--scenario", "-s", default=None,
                        help="Apply a scenario before export (requires disruption_scenarios)")
    parser.add_argument("--dual", action="store_true",
                        help="Export dual graph connections")
    parser.add_argument("--baseline", action="store_true",
                        help="Export baseline (no disruption)")
    parser.add_argument("--pois", action="store_true",
                        help="Also export POIs")
    parser.add_argument("--lts", action="store_true",
                        help="Also export low-stress cycling network")
    parser.add_argument("--lts-max", type=int, default=2,
                        help="Max LTS for low-stress export (default: 2)")

    args = parser.parse_args()
    city = args.city.lower().strip()
    out_dir = os.path.join(args.output, city)
    os.makedirs(out_dir, exist_ok=True)

    exporter = QGISExporter(args.neo4jURL, args.neo4juser, args.neo4jpwd)

    try:
        # Optionally apply a disruption scenario first
        if args.scenario:
            from disruption_engine import DisruptionEngine
            from disruption_scenarios import SCENARIOS, DUAL_SCENARIOS
            engine = DisruptionEngine(args.neo4jURL, args.neo4juser, args.neo4jpwd)
            engine.reset_all()
            if args.scenario in SCENARIOS:
                SCENARIOS[args.scenario]["func"](engine, threshold_pct=10)
            elif args.scenario in DUAL_SCENARIOS:
                engine.dual_reset_all()
                DUAL_SCENARIOS[args.scenario]["func"](engine, threshold_pct=10)
            engine.close()
            print("  Applied scenario: {}".format(args.scenario))

        # Export
        suffix = "_{}".format(args.scenario) if args.scenario else ""
        suffix += "_baseline" if args.baseline else ""

        print("\n=== QGIS Export: {} ===".format(city.upper()))

        exporter.export_primal_edges(
            os.path.join(out_dir, "{}_edges{}.csv".format(city, suffix)),
            include_disrupted=not args.baseline
        )

        if args.dual:
            exporter.export_dual_connections(
                os.path.join(out_dir, "{}_dual{}.csv".format(city, suffix))
            )

        if args.pois:
            exporter.export_pois(
                os.path.join(out_dir, "{}_pois.csv".format(city))
            )

        if args.lts:
            exporter.export_lts_network(
                os.path.join(out_dir, "{}_lts_network.csv".format(city)),
                max_lts=args.lts_max
            )

        print("\n  Done! Import CSVs into QGIS as 'Delimited Text Layer' with WKT geometry.")

    finally:
        exporter.close()


if __name__ == "__main__":
    main()
