import argparse
from neo4j import GraphDatabase
import time


class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password),
            notifications_min_severity="OFF"
        )

    def close(self):
        self.driver.close()

    def creation_graph(self):
        with self.driver.session() as session:
            # Create indexes FIRST so all subsequent queries are fast
            session.run("CREATE INDEX IF NOT EXISTS FOR (r:RoadOsm) ON (r.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR ()-[r:ROUTE]-() ON (r.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR ()-[r:ROUTE]-() ON (r.status)")
            session.run("CREATE INDEX IF NOT EXISTS FOR ()-[r:ROUTE]-() ON (r.junction)")
            print("  Indexes ensured: RoadOsm.name, ROUTE.name, ROUTE.status, ROUTE.junction")

        self._run_step("Step 1", self._step1_create_nodes)
        self._run_step("Step 2", self._step2_aggregate_properties)
        self._run_step("Step 3", self._step3_dominant_types)
        self._run_step("Step 4", self._step4_connections)
        self._run_step("Step 5", self._step5_roundabouts)
        self._run_step("Step 6", self._step6_unnamed, use_apoc=True)

    def _run_step(self, label, fn, use_apoc=False):
        """Run each step in its own transaction to avoid huge tx logs."""
        t0 = time.time()
        with self.driver.session() as session:
            if use_apoc:
                fn(session)
            else:
                session.execute_write(fn)
        print(f"  {label} done ({time.time()-t0:.1f}s)")

    @staticmethod
    def _step1_create_nodes(tx):
        # Step 1: Create RoadOsm nodes based on ROAD NAME
        tx.run("""
            MATCH (m:RoadJunction)-[r:ROUTE {status: 'active'}]->(n:RoadJunction)
            WHERE r.name IS NOT NULL
            WITH DISTINCT r.name AS road_name
            MERGE (road:RoadOsm {name: road_name})
            SET road.status = 'active'
        """)

    @staticmethod
    def _step2_aggregate_properties(tx):
        # Step 2: Store osmway_ids, distance, and modal properties
        tx.run("""
            MATCH (m:RoadJunction)-[r:ROUTE {status: 'active'}]->(n:RoadJunction)
            WHERE r.name IS NOT NULL
            WITH r.name AS road_name,
                 sum(r.distance) AS total_dist,
                 collect(DISTINCT r.osmway_id) AS osmway_ids,
                 collect(r.driveable) AS drv_flags,
                 collect(r.bike_allowed) AS bike_flags,
                 collect(r.lts) AS lts_vals,
                 collect(r.bicycle_class) AS bclass_vals,
                 collect(r.speed) AS speed_vals,
                 collect(r.lanes) AS lane_vals,
                 collect(r.route_type) AS rt_vals,
                 collect(r.on_bus_route) AS bus_flags,
                 count(r) AS seg_count
            MATCH (d:RoadOsm {name: road_name})
            SET d.distance       = total_dist,
                d.osmway_ids     = osmway_ids,
                d.segment_count  = seg_count,
                d.driveable      = ANY(f IN drv_flags WHERE f = true),
                d.bike_allowed   = ANY(f IN bike_flags WHERE f = true),
                d.on_bus_route   = ANY(f IN bus_flags WHERE f = true),
                d.avg_speed      = toFloat(reduce(s = 0, x IN [v IN speed_vals WHERE v IS NOT NULL] | s + x))
                                   / CASE WHEN size([v IN speed_vals WHERE v IS NOT NULL]) > 0
                                          THEN size([v IN speed_vals WHERE v IS NOT NULL])
                                          ELSE 1 END,
                d.avg_lanes      = toFloat(reduce(s = 0, x IN [v IN lane_vals WHERE v IS NOT NULL] | s + x))
                                   / CASE WHEN size([v IN lane_vals WHERE v IS NOT NULL]) > 0
                                          THEN size([v IN lane_vals WHERE v IS NOT NULL])
                                          ELSE 1 END,
                d.min_lts        = reduce(m = 999, x IN [v IN lts_vals WHERE v IS NOT NULL] |
                                          CASE WHEN x < m THEN x ELSE m END),
                d.max_lts        = reduce(m = 0, x IN [v IN lts_vals WHERE v IS NOT NULL] |
                                          CASE WHEN x > m THEN x ELSE m END),
                d.dominant_bicycle_class = reduce(m = 999, x IN [v IN bclass_vals WHERE v IS NOT NULL] |
                                                  CASE WHEN x < m THEN x ELSE m END)
        """)
        # Fix sentinels: if no bike segments, set LTS/class to null;
        # if no speed/lane data, set averages to null.
        tx.run("""
            MATCH (d:RoadOsm)
            WHERE d.min_lts = 999
            SET d.min_lts = null, d.max_lts = null, d.dominant_bicycle_class = null
        """)
        tx.run("""
            MATCH (d:RoadOsm)
            WHERE d.avg_speed = 0.0
            SET d.avg_speed = null
        """)
        tx.run("""
            MATCH (d:RoadOsm)
            WHERE d.avg_lanes = 0.0
            SET d.avg_lanes = null
        """)

    @staticmethod
    def _step3_dominant_types(tx):
        # Step 3: Compute dominant route_type and highway type
        tx.run("""
            MATCH (:RoadJunction)-[r:ROUTE {status: 'active'}]->(:RoadJunction)
            WHERE r.name IS NOT NULL
            WITH r.name AS road_name, r.route_type AS rt, count(*) AS cnt
            ORDER BY road_name, cnt DESC
            WITH road_name, collect(rt)[0] AS dominant_rt
            MATCH (d:RoadOsm {name: road_name})
            SET d.route_type = dominant_rt
        """)
        tx.run("""
            MATCH (:RoadJunction)-[r:ROUTE {status: 'active'}]->(:RoadJunction)
            WHERE r.name IS NOT NULL AND r.highway IS NOT NULL
            WITH r.name AS road_name, r.highway AS hw, count(*) AS cnt
            ORDER BY road_name, cnt DESC
            WITH road_name, collect(hw)[0] AS dominant_hw
            MATCH (d:RoadOsm {name: road_name})
            SET d.highway = dominant_hw
        """)

    @staticmethod
    def _step4_connections(tx):
        # Step 4: BIDIRECTIONAL connections between roads sharing a junction.
        # OPTIMIZED: direct junction-based join instead of UNWIND pattern.
        tx.run("""
            MATCH (:RoadJunction)-[r1:ROUTE {status: 'active'}]->(m:RoadJunction)<-[r2:ROUTE {status: 'active'}]-(:RoadJunction)
            WHERE r1.name IS NOT NULL AND r2.name IS NOT NULL AND r1.name <> r2.name
            WITH r1.name AS name1, r2.name AS name2,
                 collect(DISTINCT m.osmnode_id) AS junction_ids,
                 collect(DISTINCT m.location)   AS location_ids,
                 collect(r1.driveable) AS drv1, collect(r2.driveable) AS drv2,
                 collect(r1.bike_allowed) AS bike1, collect(r2.bike_allowed) AS bike2
            WITH name1, name2, junction_ids, location_ids,
                 ANY(d IN drv1 WHERE d = true) AND ANY(d IN drv2 WHERE d = true) AS car_trav,
                 ANY(b IN bike1 WHERE b = true) AND ANY(b IN bike2 WHERE b = true) AS bike_trav
            MATCH (ro1:RoadOsm {name: name1}), (ro2:RoadOsm {name: name2})
            MERGE (ro1)-[c:CONNECTED]->(ro2)
            SET c.junctions        = junction_ids,
                c.locations        = location_ids,
                c.car_traversable  = car_trav,
                c.bike_traversable = bike_trav
        """)

    @staticmethod
    def _step5_roundabouts(tx):
        # Step 5: Connect RoadOsm nodes through roundabouts (bidirectional)
        # Reduced path length from *1..30 to *1..10 (roundabouts rarely > 10 segments)
        tx.run("""
            MATCH (x:RoadJunction)-[r1:ROUTE {status: 'active'}]->(rbNode:RoadJunction)
            WHERE r1.name IS NOT NULL AND r1.junction IS NULL
            MATCH (rbNode)-[:ROUTE*1..15 {junction: 'roundabout', status: 'active'}]->(rbNode2:RoadJunction)
            MATCH (rbNode2)-[r2:ROUTE {status: 'active'}]->(y:RoadJunction)
            WHERE r2.name IS NOT NULL AND r2.junction IS NULL AND r1.name <> r2.name
            WITH r1.name AS road1, r2.name AS road2,
                 collect(r1.driveable) AS r1_drv_all, collect(r2.driveable) AS r2_drv_all,
                 collect(r1.bike_allowed) AS r1_bike_all, collect(r2.bike_allowed) AS r2_bike_all
            WITH DISTINCT road1, road2,
                 ANY(d IN r1_drv_all WHERE d = true) AND ANY(d IN r2_drv_all WHERE d = true) AS car_trav,
                 ANY(b IN r1_bike_all WHERE b = true) AND ANY(b IN r2_bike_all WHERE b = true) AS bike_trav
            MATCH (ro1:RoadOsm {name: road1}), (ro2:RoadOsm {name: road2})
            MERGE (ro1)-[c:CONNECTED]->(ro2)
            ON CREATE SET c.via = 'roundabout',
                          c.car_traversable = car_trav,
                          c.bike_traversable = bike_trav
        """)

    @staticmethod
    def _step6_unnamed(session):
        # Step 6: Connect RoadOsm nodes through unnamed segments (batched)
        # Uses apoc.periodic.iterate to avoid OOM on large graphs.
        # Outer query: find each named ROUTE edge that borders an unnamed segment.
        # Inner query: for each such edge, traverse unnamed segments up to 15 hops
        # and connect the RoadOsm nodes at each end.
        result = session.run("""
            CALL apoc.periodic.iterate(
                "MATCH (x:RoadJunction)-[r1:ROUTE {status: 'active'}]->(mid:RoadJunction)
                 WHERE r1.name IS NOT NULL AND r1.junction IS NULL
                 RETURN r1.name AS r1name, mid,
                        r1.driveable AS r1drv, r1.bike_allowed AS r1bike",

                "MATCH path = (mid)-[:ROUTE*1..15]->(mid2:RoadJunction)
                 WHERE ALL(r IN relationships(path)
                           WHERE r.status = 'active'
                             AND r.name IS NULL
                             AND r.junction IS NULL)
                 MATCH (mid2)-[r2:ROUTE {status: 'active'}]->(y:RoadJunction)
                 WHERE r2.name IS NOT NULL AND r2.junction IS NULL AND r1name <> r2.name
                 WITH r1name AS road1, r2.name AS road2,
                      r1drv AS r1d, r2.driveable AS r2d,
                      r1bike AS r1b, r2.bike_allowed AS r2b
                 WITH DISTINCT road1, road2,
                      (r1d = true AND r2d = true) AS car_trav,
                      (r1b = true AND r2b = true) AS bike_trav
                 MATCH (ro1:RoadOsm {name: road1}), (ro2:RoadOsm {name: road2})
                 MERGE (ro1)-[c:CONNECTED]->(ro2)
                 ON CREATE SET c.via = 'unnamed',
                               c.car_traversable = car_trav,
                               c.bike_traversable = bike_trav",

                {batchSize: 100, iterateList: true, parallel: false}
            )
            YIELD batches, total, errorMessages
            RETURN batches, total, errorMessages
        """).single()
        if result:
            print(f"    Step 6 batched: {result['total']} items in {result['batches']} batches")
            if result['errorMessages']:
                print(f"    Errors: {result['errorMessages']}")

    # ------------------------------------------------------------------
    #  CONNECT POIs to RoadOsm nodes
    #  Each POI's nearest RoadJunction links to ROUTE edges with names.
    #  Those names map to RoadOsm nodes. We create NEAR_ROAD relationships
    #  from POI -> RoadOsm, preserving modal info.
    # ------------------------------------------------------------------
    def connect_pois_to_roads(self):
        with self.driver.session() as session:
            session.execute_write(self._connect_pois_to_roads)

    @staticmethod
    def _connect_pois_to_roads(tx):
        # --- 7a: Way POIs (OSMWay:POI) via their member OSMNodes ---
        # Path: OSMWay:POI <-[:PART_OF]- OSMNode -[:NEAR_CAR]-> RoadJunction -[:ROUTE]-> ...
        tx.run("""
            CALL apoc.periodic.iterate(
                "MATCH (poi:OSMWay:POI)
                 RETURN poi",
                "MATCH (osmn:OSMNode)-[:PART_OF]->(poi)
                 MATCH (osmn)-[:NEAR_CAR]->(rj:RoadJunction)
                 MATCH (rj)-[r:ROUTE {status: 'active'}]->(:RoadJunction)
                 WHERE r.name IS NOT NULL
                 WITH poi, DISTINCT r.name AS rn
                 MATCH (ro:RoadOsm {name: rn})
                 MERGE (poi)-[:NEAR_ROAD {mode: 'car'}]->(ro)",
                {batchSize:200, iterateList:true, parallel:false}
            )
            YIELD batches, total
            RETURN batches, total
        """).consume()

        tx.run("""
            CALL apoc.periodic.iterate(
                "MATCH (poi:OSMWay:POI)
                 RETURN poi",
                "MATCH (osmn:OSMNode)-[:PART_OF]->(poi)
                 MATCH (osmn)-[:NEAR_BIKE]->(rj:RoadJunction)
                 MATCH (rj)-[r:ROUTE {status: 'active'}]->(:RoadJunction)
                 WHERE r.name IS NOT NULL AND r.bike_allowed = true
                 WITH poi, DISTINCT r.name AS rn
                 MATCH (ro:RoadOsm {name: rn})
                 MERGE (poi)-[:NEAR_ROAD {mode: 'bike'}]->(ro)",
                {batchSize:200, iterateList:true, parallel:false}
            )
            YIELD batches, total
            RETURN batches, total
        """).consume()
        print("  Step 7a: Connected Way POIs to RoadOsm nodes")

        # --- 7b: Node POIs (OSMNode:POI) directly via NEAR_CAR / NEAR_BIKE ---
        tx.run("""
            CALL apoc.periodic.iterate(
                "MATCH (poi:OSMNode:POI)
                 RETURN poi",
                "MATCH (poi)-[:NEAR_CAR]->(rj:RoadJunction)
                 MATCH (rj)-[r:ROUTE {status: 'active'}]->(:RoadJunction)
                 WHERE r.name IS NOT NULL
                 WITH poi, DISTINCT r.name AS rn
                 MATCH (ro:RoadOsm {name: rn})
                 MERGE (poi)-[:NEAR_ROAD {mode: 'car'}]->(ro)",
                {batchSize:200, iterateList:true, parallel:false}
            )
            YIELD batches, total
            RETURN batches, total
        """).consume()

        tx.run("""
            CALL apoc.periodic.iterate(
                "MATCH (poi:OSMNode:POI)
                 RETURN poi",
                "MATCH (poi)-[:NEAR_BIKE]->(rj:RoadJunction)
                 MATCH (rj)-[r:ROUTE {status: 'active'}]->(:RoadJunction)
                 WHERE r.name IS NOT NULL AND r.bike_allowed = true
                 WITH poi, DISTINCT r.name AS rn
                 MATCH (ro:RoadOsm {name: rn})
                 MERGE (poi)-[:NEAR_ROAD {mode: 'bike'}]->(ro)",
                {batchSize:200, iterateList:true, parallel:false}
            )
            YIELD batches, total
            RETURN batches, total
        """).consume()
        print("  Step 7b: Connected Node POIs to RoadOsm nodes")

    def set_index(self):
        """Index already created at start of creation_graph — kept for compatibility."""
        pass


def add_options():
    parser = argparse.ArgumentParser(description='Creation of name-based dual graph.')
    parser.add_argument('--neo4jURL', '-n', dest='neo4jURL', type=str,
                        help='Address of local neo4j instance (e.g. neo4j://localhost:7687)',
                        required=True)
    parser.add_argument('--neo4juser', '-u', dest='neo4juser', type=str,
                        help='Neo4j username', required=True)
    parser.add_argument('--neo4jpwd', '-p', dest='neo4jpwd', type=str,
                        help='Neo4j password', required=True)
    return parser


def main(args=None):
    start_time = time.time()
    argParser = add_options()
    options = argParser.parse_args(args=args)
    greeter = App(options.neo4jURL, options.neo4juser, options.neo4jpwd)

    print("\n=== Creating Dual Graph (Name-based, with modal attributes) ===\n")
    greeter.creation_graph()
    greeter.connect_pois_to_roads()
    greeter.set_index()
    greeter.close()

    elapsed = time.time() - start_time
    print(f"\nDual graph created successfully! ({elapsed:.2f}s)")
    return 0


if __name__ == '__main__':
    main()
