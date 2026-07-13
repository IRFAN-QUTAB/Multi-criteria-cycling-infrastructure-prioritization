import argparse
import time
from neo4j import GraphDatabase


class CentralityComputer:

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

    # ------------------------------------------------------------------
    #  GDS PROJECTED GRAPH MANAGEMENT
    # ------------------------------------------------------------------

    def _drop_if_exists(self, graph_name):
        """Drop a GDS projected graph if it exists."""
        try:
            self._run("CALL gds.graph.drop($name, false)", name=graph_name)
        except Exception:
            pass

    def _create_projection(self, graph_name, mode):
        """Create a GDS Cypher projection filtered by mode.
        
        mode='car'  -> only driveable=true edges
        mode='bike' -> only bike_allowed=true edges
        """
        self._drop_if_exists(graph_name)

        if mode == "car":
            edge_filter = "r.driveable = true"
        elif mode == "bike":
            edge_filter = "r.bike_allowed = true"
        else:
            edge_filter = "true"

        node_query = "MATCH (n:RoadJunction) RETURN id(n) AS id"
        rel_query = (
            "MATCH (a:RoadJunction)-[r:ROUTE]->(b:RoadJunction) "
            "WHERE {} AND r.status = 'active' "
            "RETURN id(a) AS source, id(b) AS target, "
            "coalesce(r.distance, 1.0) AS distance"
        ).format(edge_filter)

        self._run(
            "CALL gds.graph.project.cypher($name, $nq, $rq,"
            "{validateRelationships: false})",
            name=graph_name, nq=node_query, rq=rel_query
        )
        print("  Created projection: {}".format(graph_name))

    # ------------------------------------------------------------------
    #  CENTRALITY ALGORITHMS
    # ------------------------------------------------------------------

    def _compute_betweenness(self, graph_name, write_property):
        """Betweenness Centrality - Brandes algorithm."""
        q = ("CALL gds.betweenness.write($name, {writeProperty: $prop}) "
             "YIELD nodePropertiesWritten, computeMillis")
        r = self._run(q, name=graph_name, prop=write_property)
        if r:
            print("    BC -> {} nodes, {}ms".format(
                r[0]["nodePropertiesWritten"], r[0]["computeMillis"]))

    def _compute_pagerank(self, graph_name, write_property):
        """PageRank with damping=0.85, max 20 iterations."""
        q = ("CALL gds.pageRank.write($name, {"
             "writeProperty: $prop, maxIterations: 20, dampingFactor: 0.85"
             "}) YIELD nodePropertiesWritten, computeMillis")
        r = self._run(q, name=graph_name, prop=write_property)
        if r:
            print("    PageRank -> {} nodes, {}ms".format(
                r[0]["nodePropertiesWritten"], r[0]["computeMillis"]))

    def _compute_degree(self, graph_name, write_property):
        """Degree Centrality."""
        q = ("CALL gds.degree.write($name, {writeProperty: $prop}) "
             "YIELD nodePropertiesWritten, computeMillis")
        r = self._run(q, name=graph_name, prop=write_property)
        if r:
            print("    Degree -> {} nodes, {}ms".format(
                r[0]["nodePropertiesWritten"], r[0]["computeMillis"]))

    def _compute_harmonic(self, graph_name, write_property):
        """Harmonic Closeness Centrality.
        
        Tries gds.closeness.harmonic first (GDS 2.x production),
        falls back to gds.beta.closeness.harmonic (older GDS).
        """
        for proc in ["gds.closeness.harmonic.write",
                      "gds.beta.closeness.harmonic.write"]:
            try:
                q = ("CALL {}($name, {{writeProperty: $prop}}) "
                     "YIELD nodePropertiesWritten, computeMillis").format(proc)
                r = self._run(q, name=graph_name, prop=write_property)
                if r:
                    print("    Harmonic Closeness -> {} nodes, {}ms (via {})".format(
                        r[0]["nodePropertiesWritten"], r[0]["computeMillis"], proc))
                return
            except Exception as e:
                continue

        # Last fallback: regular closeness
        try:
            q = ("CALL gds.closeness.write($name, {writeProperty: $prop}) "
                 "YIELD nodePropertiesWritten, computeMillis")
            r = self._run(q, name=graph_name, prop=write_property)
            if r:
                print("    Closeness (regular) -> {} nodes, {}ms".format(
                    r[0]["nodePropertiesWritten"], r[0]["computeMillis"]))
        except Exception as e:
            print("    WARNING: Closeness failed ({}). Skipping.".format(e))

    # ------------------------------------------------------------------
    #  COMPUTE ALL FOR ONE MODE
    # ------------------------------------------------------------------

    def compute_for_mode(self, mode):
        """Compute all 4 centrality measures for car or bike subgraph."""
        graph_name = "{}_centrality".format(mode)
        prefix = mode  # car_bc, bike_bc, etc.

        print("\n  === {} subgraph ===".format(mode.upper()))
        t0 = time.time()

        self._create_projection(graph_name, mode)
        self._compute_betweenness(graph_name, "{}_bc".format(prefix))
        self._compute_pagerank(graph_name, "{}_pagerank".format(prefix))
        self._compute_degree(graph_name, "{}_degree".format(prefix))
        self._compute_harmonic(graph_name, "{}_harmonic".format(prefix))
        self._drop_if_exists(graph_name)

        print("  {} done in {:.1f}s".format(mode.upper(), time.time() - t0))

    # ------------------------------------------------------------------
    #  AGGREGATE TO ROAD LEVEL (RoadOsm)
    # ------------------------------------------------------------------

    def aggregate_to_roads(self):
        """Aggregate junction-level centrality to named-road level.
        
        For each RoadOsm node, compute avg/max centrality of junctions
        on that road's ROUTE edges.
        """
        print("\n  === Aggregating to road level ===")

        for mode in ["car", "bike"]:
            q = """
            MATCH (a:RoadJunction)-[r:ROUTE]->(b:RoadJunction)
            WHERE r.name IS NOT NULL AND r.status = 'active'
            WITH r.name AS road_name,
                 collect(DISTINCT a) + collect(DISTINCT b) AS junctions
            UNWIND junctions AS j
            WITH road_name,
                 avg(j.{m}_bc) AS avg_bc,
                 max(j.{m}_bc) AS max_bc,
                 avg(j.{m}_pagerank) AS avg_pr,
                 max(j.{m}_pagerank) AS max_pr,
                 avg(j.{m}_degree) AS avg_deg,
                 max(j.{m}_degree) AS max_deg,
                 avg(j.{m}_harmonic) AS avg_hc,
                 max(j.{m}_harmonic) AS max_hc
            MATCH (ro:RoadOsm {{name: road_name}})
            SET ro.{m}_bc_avg = avg_bc,
                ro.{m}_bc_max = max_bc,
                ro.{m}_pagerank_avg = avg_pr,
                ro.{m}_pagerank_max = max_pr,
                ro.{m}_degree_avg = avg_deg,
                ro.{m}_degree_max = max_deg,
                ro.{m}_harmonic_avg = avg_hc,
                ro.{m}_harmonic_max = max_hc
            RETURN count(ro) AS updated
            """.format(m=mode)
            result = self._run(q)
            n = result[0]["updated"] if result else 0
            print("    {} centrality aggregated to {} roads".format(mode, n))

    # ------------------------------------------------------------------
    #  DUAL GRAPH CENTRALITY (RoadOsm + CONNECTED)
    # ------------------------------------------------------------------

    def _create_dual_projection(self, graph_name):
        """Project dual graph: RoadOsm nodes, CONNECTED edges with weight."""
        self._drop_if_exists(graph_name)

        node_query = "MATCH (n:RoadOsm) RETURN id(n) AS id"
        rel_query = (
            "MATCH (a:RoadOsm)-[r:CONNECTED]->(b:RoadOsm) "
            "RETURN id(a) AS source, id(b) AS target, "
            "coalesce(r.weight, 1.0) AS weight"
        )

        self._run(
            "CALL gds.graph.project.cypher($name, $nq, $rq,"
            "{validateRelationships: false})",
            name=graph_name, nq=node_query, rq=rel_query
        )

    def compute_dual_centrality(self):
        """All centrality measures on dual graph (RoadOsm + CONNECTED).

        Gives road-level importance directly. Properties written:
          dual_pagerank, dual_bc, dual_degree, dual_harmonic
        """
        print("\n  === Dual Graph Centrality ===")
        graph_name = "dual_centrality"
        self._create_dual_projection(graph_name)

        # PageRank (weighted)
        q = ("CALL gds.pageRank.write($name, {"
             "writeProperty: 'dual_pagerank', "
             "maxIterations: 20, dampingFactor: 0.85, "
             "relationshipWeightProperty: 'weight'"
             "}) YIELD nodePropertiesWritten, computeMillis")
        r = self._run(q, name=graph_name)
        if r:
            print("    Dual PageRank -> {} nodes, {}ms".format(
                r[0]["nodePropertiesWritten"], r[0]["computeMillis"]))

        # Betweenness
        q = ("CALL gds.betweenness.write($name, {"
             "writeProperty: 'dual_bc'"
             "}) YIELD nodePropertiesWritten, computeMillis")
        r = self._run(q, name=graph_name)
        if r:
            print("    Dual BC -> {} nodes, {}ms".format(
                r[0]["nodePropertiesWritten"], r[0]["computeMillis"]))

        # Degree
        q = ("CALL gds.degree.write($name, {"
             "writeProperty: 'dual_degree'"
             "}) YIELD nodePropertiesWritten, computeMillis")
        r = self._run(q, name=graph_name)
        if r:
            print("    Dual Degree -> {} nodes, {}ms".format(
                r[0]["nodePropertiesWritten"], r[0]["computeMillis"]))

        # Harmonic Closeness
        for proc in ["gds.closeness.harmonic.write",
                      "gds.beta.closeness.harmonic.write"]:
            try:
                q = ("CALL {}($name, {{writeProperty: 'dual_harmonic'}}) "
                     "YIELD nodePropertiesWritten, computeMillis").format(proc)
                r = self._run(q, name=graph_name)
                if r:
                    print("    Dual Harmonic -> {} nodes, {}ms (via {})".format(
                        r[0]["nodePropertiesWritten"], r[0]["computeMillis"], proc))
                break
            except Exception:
                continue

        self._drop_if_exists(graph_name)

        # Print top roads
        for prop, label in [("dual_pagerank", "PageRank"),
                            ("dual_bc", "Betweenness"),
                            ("dual_degree", "Degree"),
                            ("dual_harmonic", "Harmonic")]:
            top_q = """
            MATCH (ro:RoadOsm)
            WHERE ro.{} IS NOT NULL AND ro.name IS NOT NULL
            RETURN ro.name AS name, ro.{} AS score
            ORDER BY ro.{} DESC
            LIMIT 10
            """.format(prop, prop, prop)
            results = self._run(top_q)
            print("\n    Top 10 by dual {}:".format(label))
            for i, r in enumerate(results, 1):
                print("      {:2d}. {:40s} {:.6f}".format(
                    i, (r["name"] or "?")[:40], r["score"] or 0))

    # ------------------------------------------------------------------
    #  PRINT TOP ROADS
    # ------------------------------------------------------------------

    def print_top_roads(self, n=20):
        """Print top N roads by each centrality metric."""
        metrics = [
            ("car_bc_avg", "Car Betweenness (avg)"),
            ("bike_bc_avg", "Bike Betweenness (avg)"),
            ("car_pagerank_avg", "Car PageRank (avg)"),
            ("bike_pagerank_avg", "Bike PageRank (avg)"),
            ("car_harmonic_avg", "Car Harmonic Closeness (avg)"),
            ("bike_harmonic_avg", "Bike Harmonic Closeness (avg)"),
        ]

        for prop, label in metrics:
            print("\n  Top {} roads by {}:".format(n, label))
            q = """
            MATCH (ro:RoadOsm)
            WHERE ro.{} IS NOT NULL
            RETURN ro.name AS name, ro.{} AS score,
                   ro.highway AS highway, ro.driveable AS driveable,
                   ro.bike_allowed AS bike_allowed
            ORDER BY ro.{} DESC
            LIMIT $n
            """.format(prop, prop, prop)
            results = self._run(q, n=n)
            for i, r in enumerate(results, 1):
                print("    {:2d}. {:40s} score={:.6f}  hw={} car={} bike={}".format(
                    i, (r["name"] or "?")[:40], r["score"] or 0,
                    r["highway"] or "?",
                    r["driveable"], r["bike_allowed"]))

    # ------------------------------------------------------------------
    #  MAIN ENTRY
    # ------------------------------------------------------------------

    def compute_all(self, top_n=20):
        """Run the full centrality pipeline."""
        print("\n" + "=" * 60)
        print("  CENTRALITY COMPUTATION (Neo4j GDS)")
        print("=" * 60)

        t0 = time.time()
        self.compute_for_mode("car")
        self.compute_for_mode("bike")
        self.aggregate_to_roads()
        self.compute_dual_centrality()
        self.print_top_roads(top_n)

        print("\n  Total centrality computation: {:.1f}s".format(time.time() - t0))


def main():
    parser = argparse.ArgumentParser(
        description="Compute real centrality measures via Neo4j GDS"
    )
    parser.add_argument("--neo4jURL", "-n", required=True)
    parser.add_argument("--neo4juser", "-u", required=True)
    parser.add_argument("--neo4jpwd", "-p", required=True)
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N roads per metric (default: 20)")

    args = parser.parse_args()

    computer = CentralityComputer(args.neo4jURL, args.neo4juser, args.neo4jpwd)
    try:
        computer.compute_all(top_n=args.top)
    finally:
        computer.close()


if __name__ == "__main__":
    main()
