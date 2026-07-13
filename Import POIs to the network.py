import overpy
import json
from neo4j import GraphDatabase
import argparse
import os
import time
import requests as req


# ============================================================================
#  OVERPASS HELPER — download with requests (reliable), parse with overpy
# ============================================================================

def overpass_query(query, max_retries=3):
    """Download with requests (avoids 406 errors), parse with overpy (convenient objects)."""
    url = "https://overpass-api.de/api/interpreter"
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'UrbanNetworkAnalysis/1.0',
    }
    for attempt in range(max_retries):
        try:
            resp = req.get(url, params={'data': query}, headers=headers, timeout=200)
            resp.raise_for_status()
            api = overpy.Overpass()
            return api.parse_json(resp.text)
        except Exception as e:
            print(f"  Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait = 15 * (attempt + 1)
                print(f"  Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise Exception(f"Overpass query failed after {max_retries} attempts")


# ============================================================================
#  NEO4J APP
# ============================================================================

class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password),
            notifications_min_severity="OFF"
        )

    def close(self):
        self.driver.close()

    def get_path(self):
        with self.driver.session() as session:
            result = session.execute_write(self._get_path)
            return result

    @staticmethod
    def _get_path(tx):
        for key in ['server.directories.neo4j_home', 'dbms.directories.neo4j_home']:
            try:
                result = tx.run(
                    "CALL dbms.listConfig() YIELD name, value "
                    "WHERE name = $key RETURN value", key=key)
                vals = result.values()
                if vals and vals[0][0]:
                    return vals
            except Exception:
                continue
        raise RuntimeError("Could not determine neo4j_home directory")

    def get_import_folder_name(self):
        with self.driver.session() as session:
            result = session.execute_write(self._get_import_folder_name)
            return result

    @staticmethod
    def _get_import_folder_name(tx):
        for key in ['server.directories.import', 'dbms.directories.import']:
            try:
                result = tx.run(
                    "CALL dbms.listConfig() YIELD name, value "
                    "WHERE name = $key RETURN value", key=key)
                vals = result.values()
                if vals and vals[0][0]:
                    return vals
            except Exception:
                continue
        raise RuntimeError("Could not determine import directory")

    # ------------------------------------------------------------------
    #  IMPORT: POI nodes (single-point amenities, shops, stops)
    #  [CHANGE 5] poi_type derived from tags, stored directly on POI node.
    # ------------------------------------------------------------------
    def import_node(self):
        with self.driver.session() as session:
            result = session.execute_write(self._import_node)
            return result

    @staticmethod
    def _import_node(tx):
        result = tx.run("""
            CALL apoc.periodic.iterate(
            "CALL apoc.load.json('nodefile.json')
            YIELD value
            UNWIND value.elements AS nodo
            return nodo",
            "MERGE (n:OSMNode:POI {osm_id: nodo.id})
            ON CREATE SET n.name=nodo.tags.name,
            n.lat=tofloat(nodo.lat),
            n.lon=tofloat(nodo.lon),
            n.geometry= 'POINT(' + nodo.lat + ' ' + nodo.lon +')',
            n.poi_type = CASE
              WHEN nodo.tags.amenity IS NOT NULL THEN nodo.tags.amenity
              WHEN nodo.tags.shop IS NOT NULL THEN nodo.tags.shop
              WHEN nodo.tags.railway IS NOT NULL THEN nodo.tags.railway
              WHEN nodo.tags.public_transport IS NOT NULL THEN nodo.tags.public_transport
              WHEN nodo.tags.highway IS NOT NULL THEN nodo.tags.highway
              ELSE 'unknown'
            END
            WITH n, nodo
            MERGE (n)-[:TAGS]->(t:Tag)
            ON CREATE SET t += nodo.tags",
            {batchSize:100, iterateList:true, parallel:false}
            )
            YIELD batches, total
            RETURN batches, total;
        """)
        return result.values()

    # ------------------------------------------------------------------
    #  IMPORT: geometry nodes that make up each way POI
    # ------------------------------------------------------------------
    def import_node_way(self):
        with self.driver.session() as session:
            result = session.execute_write(self._import_node_way)
            return result

    @staticmethod
    def _import_node_way(tx):
        result = tx.run("""
                CALL apoc.periodic.iterate(
                "CALL apoc.load.json('nodeway.json')
                YIELD value
                UNWIND value.elements AS node
                return node",
                "MERGE (n:OSMNode {osm_id: node.id})
                ON CREATE SET
                n.lat = toFloat(node.lat),
                n.lon = toFloat(node.lon),
                n.geometry = 'POINT(' + node.lat + ' ' + node.lon +')'",
                {batchSize:100, iterateList:true, parallel:false}
                )
                YIELD batches, total
                RETURN batches, total;
            """)
        return result.values()

    # ------------------------------------------------------------------
    #  IMPORT: way POIs (polygons — hospitals, schools, supermarkets)
    #  [CHANGE 5] poi_type stored on the way POI.
    # ------------------------------------------------------------------
    def import_way(self):
        with self.driver.session() as session:
            result = session.execute_write(self._import_way)
            return result

    @staticmethod
    def _import_way(tx):
        result = tx.run("""
                CALL apoc.periodic.iterate(
                "CALL apoc.load.json('wayfile.json')
                YIELD value
                UNWIND value.elements AS el
                return el",
                "MERGE (w:OSMWay:POI {osm_id: el.id})
                ON CREATE SET w.name = el.tags.name,
                w.poi_type = CASE
                  WHEN el.tags.amenity IS NOT NULL THEN el.tags.amenity
                  WHEN el.tags.shop IS NOT NULL THEN el.tags.shop
                  WHEN el.tags.railway IS NOT NULL THEN el.tags.railway
                  WHEN el.tags.public_transport IS NOT NULL THEN el.tags.public_transport
                  ELSE 'unknown'
                END
                MERGE (w)-[:TAGS]->(t:Tag)
                ON CREATE SET t += el.tags
                WITH w, el.nodes AS nodes
                UNWIND nodes AS node
                MATCH (n:OSMNode {osm_id: node})
                MERGE (n)-[:PART_OF]->(w)",
                {batchSize:100, iterateList:true, parallel:false}
                )
                YIELD batches, total
                RETURN batches, total;
            """)
        return result.values()

    def import_nodes_into_spatial_layer(self):
        try:
            with self.driver.session() as session:
                result = session.execute_write(self._import_nodes_into_spatial_layer)
                return result
        except Exception as e:
            print(f"  Spatial import skipped (plugin not installed): {e}")
            return []

    @staticmethod
    def _import_nodes_into_spatial_layer(tx):
        result = tx.run("""
            MATCH (n:OSMNode)
            CALL spatial.addNode('spatial', n) YIELD node RETURN node
        """)
        return result.values()

    # ------------------------------------------------------------------
    #  SET LOCATION on OSMNode nodes
    # ------------------------------------------------------------------
    def set_location(self):
        with self.driver.session() as session:
            result = session.execute_write(self._set_location)
            return result

    @staticmethod
    def _set_location(tx):
        result = tx.run("""
                MATCH (n:OSMNode)
                WHERE n.lat IS NOT NULL AND n.lon IS NOT NULL
                SET n.location = point({latitude: tofloat(n.lat), longitude: tofloat(n.lon)})
            """)
        return result.values()

    def set_index(self):
        with self.driver.session() as session:
            try:
                result = session.execute_write(self._set_index)
                return result
            except Exception as e:
                print(f"Index creation skipped or failed: {e}")

    @staticmethod
    def _set_index(tx):
        try:
            tx.run("CREATE INDEX IF NOT EXISTS FOR (n:OSMNode) ON (n.osm_id)")
            tx.run("CREATE INDEX IF NOT EXISTS FOR (n:POI) ON (n.poi_type)")
        except Exception as e:
            print(f"Failed to create index: {e}")
        return []

    # ------------------------------------------------------------------
    #  MARK junctions as driveable AND bike_accessible
    #  [CHANGE 2] bike_accessible flag added.
    #  [CHANGE 4] uses proper booleans (not string comparison).
    # ------------------------------------------------------------------
    def mark_accessible_roadjunctions(self):
        with self.driver.session() as session:
            session.execute_write(self._mark_accessible_roadjunctions)

    @staticmethod
    def _mark_accessible_roadjunctions(tx):
        tx.run("""
               MATCH (n:RoadJunction)
               SET n.driveable = false, n.bike_accessible = false
           """).consume()

        tx.run("""
               MATCH (n:RoadJunction)-[r:ROUTE]-()
               WHERE r.driveable = true
               SET n.driveable = true
           """).consume()

        tx.run("""
               MATCH (n:RoadJunction)-[r:ROUTE]-()
               WHERE r.bike_allowed = true
               SET n.bike_accessible = true
           """).consume()

    # ------------------------------------------------------------------
    #  CONNECT OSMNodes to nearest RoadJunctions
    #  [CHANGE 3/7/8] Two relationships: NEAR_CAR and NEAR_BIKE.
    #  Follows co-supervisor's schema: OSMNode connects to RoadJunction,
    #  OSMWay NEVER directly connects to RoadJunction.
    #  All junctions within 100m are connected (not just nearest 1).
    # ------------------------------------------------------------------
    def connect_amenity(self):
        """Connect OSMNodes to nearest RoadJunctions via NEAR_CAR and NEAR_BIKE.
        Uses session.run() directly — apoc.periodic.iterate manages its own
        transactions and must NOT be wrapped in execute_write()."""
        with self.driver.session() as session:
            # --- diagnostics before running ---
            diag = session.run("""
                MATCH (rj:RoadJunction)
                RETURN count(rj) AS total,
                       count(CASE WHEN rj.location IS NOT NULL THEN 1 END) AS has_loc,
                       count(CASE WHEN rj.driveable = true THEN 1 END) AS driveable,
                       count(CASE WHEN rj.bike_accessible = true THEN 1 END) AS bike_acc
            """).single()
            print(f"  [diag] RoadJunctions: {diag['total']}, "
                  f"with location: {diag['has_loc']}, "
                  f"driveable: {diag['driveable']}, "
                  f"bike_accessible: {diag['bike_acc']}")

            osm_diag = session.run("""
                MATCH (n:OSMNode)
                RETURN count(n) AS total,
                       count(CASE WHEN n.lat IS NOT NULL AND n.lon IS NOT NULL THEN 1 END) AS has_coords
            """).single()
            print(f"  [diag] OSMNodes: {osm_diag['total']}, with lat/lon: {osm_diag['has_coords']}")

            # ============ NEAR_CAR ============

            # Step 1: All driveable junctions within 100m
            r = session.run("""
                CALL apoc.periodic.iterate(
                    "MATCH (osmn:OSMNode)
                     WHERE NOT (osmn)-[:NEAR_CAR]->(:RoadJunction)
                       AND osmn.lat IS NOT NULL AND osmn.lon IS NOT NULL
                     RETURN osmn",
                    "WITH osmn, point({latitude: toFloat(osmn.lat), longitude: toFloat(osmn.lon)}) AS nodeLoc
                     MATCH (rj:RoadJunction)
                     WHERE rj.driveable = true AND rj.location IS NOT NULL AND point.distance(rj.location, nodeLoc) < 100
                     WITH osmn, rj, point.distance(rj.location, nodeLoc) AS dist
                     MERGE (osmn)-[r:NEAR_CAR]->(rj)
                       ON CREATE SET r.distance = dist, r.status = 'driveable_within_100m'",
                    {batchSize:500, iterateList:true, parallel:false}
                )
                YIELD batches, total, errorMessages
                RETURN batches, total, errorMessages
            """).single()
            print(f"  NEAR_CAR step1: batches={r['batches']}, total={r['total']}, errors={r['errorMessages']}")

            # Step 2: If no driveable within 100m → nearest non-driveable
            r = session.run("""
                CALL apoc.periodic.iterate(
                    "MATCH (osmn:OSMNode)
                     WHERE NOT (osmn)-[:NEAR_CAR]->(:RoadJunction)
                       AND osmn.lat IS NOT NULL AND osmn.lon IS NOT NULL
                     RETURN osmn",
                    "WITH osmn, point({latitude: toFloat(osmn.lat), longitude: toFloat(osmn.lon)}) AS osmLoc
                     CALL {
                       WITH osmn, osmLoc
                       MATCH (rj:RoadJunction)
                       WHERE rj.driveable = false AND rj.location IS NOT NULL
                       RETURN rj, point.distance(osmLoc, rj.location) AS dist
                       ORDER BY dist ASC
                       LIMIT 1
                     }
                     MERGE (osmn)-[r:NEAR_CAR]->(rj)
                       ON CREATE SET r.distance = dist, r.status = 'nearest_non_driveable'",
                    {batchSize:500, iterateList:true, parallel:false}
                )
                YIELD batches, total, errorMessages
                RETURN batches, total, errorMessages
            """).single()
            print(f"  NEAR_CAR step2: batches={r['batches']}, total={r['total']}, errors={r['errorMessages']}")

            # Step 3: Way POIs — at least 1 node connected to driveable
            r = session.run("""
                CALL apoc.periodic.iterate(
                    "MATCH (osmn:OSMNode)-[:NEAR_CAR]->(nonRJ:RoadJunction)
                     WHERE nonRJ.driveable = false
                       AND NOT (osmn)-[:NEAR_CAR]->(:RoadJunction {driveable: true})
                       AND osmn.lat IS NOT NULL AND osmn.lon IS NOT NULL
                     RETURN osmn",
                    "WITH osmn, point({latitude: toFloat(osmn.lat), longitude: toFloat(osmn.lon)}) AS osmLoc
                     CALL {
                       WITH osmn, osmLoc
                       MATCH (drvRJ:RoadJunction)
                       WHERE drvRJ.driveable = true AND drvRJ.location IS NOT NULL
                       RETURN drvRJ, point.distance(osmLoc, drvRJ.location) AS dist
                       ORDER BY dist ASC
                       LIMIT 1
                     }
                     MERGE (osmn)-[r:NEAR_CAR]->(drvRJ)
                       ON CREATE SET r.distance = dist, r.status = 'nearest_driveable_from_non_driveable'",
                    {batchSize:500, iterateList:true, parallel:false}
                )
                YIELD batches, total, errorMessages
                RETURN batches, total, errorMessages
            """).single()
            print(f"  NEAR_CAR step3: batches={r['batches']}, total={r['total']}, errors={r['errorMessages']}")

            # ============ NEAR_BIKE ============

            # Step 1: All bike_accessible junctions within 100m
            r = session.run("""
                CALL apoc.periodic.iterate(
                    "MATCH (osmn:OSMNode)
                     WHERE NOT (osmn)-[:NEAR_BIKE]->(:RoadJunction)
                       AND osmn.lat IS NOT NULL AND osmn.lon IS NOT NULL
                     RETURN osmn",
                    "WITH osmn, point({latitude: toFloat(osmn.lat), longitude: toFloat(osmn.lon)}) AS nodeLoc
                     MATCH (rj:RoadJunction)
                     WHERE rj.bike_accessible = true AND rj.location IS NOT NULL AND point.distance(rj.location, nodeLoc) < 100
                     With osmn, rj, point.distance(rj.location, nodeLoc) AS dist
                     MERGE (osmn)-[r:NEAR_BIKE]->(rj)
                       ON CREATE SET r.distance = dist, r.status = 'bike_accessible_within_100m'",
                    {batchSize:500, iterateList:true, parallel:false}
                )
                YIELD batches, total, errorMessages
                RETURN batches, total, errorMessages
            """).single()
            print(f"  NEAR_BIKE step1: batches={r['batches']}, total={r['total']}, errors={r['errorMessages']}")

            # Step 2: If no bike_accessible within 100m → nearest non-bike
            r = session.run("""
                CALL apoc.periodic.iterate(
                    "MATCH (osmn:OSMNode)
                     WHERE NOT (osmn)-[:NEAR_BIKE]->(:RoadJunction)
                       AND osmn.lat IS NOT NULL AND osmn.lon IS NOT NULL
                     RETURN osmn",
                    "WITH osmn, point({latitude: toFloat(osmn.lat), longitude: toFloat(osmn.lon)}) AS osmLoc
                     CALL {
                       WITH osmn, osmLoc
                       MATCH (rj:RoadJunction)
                       WHERE rj.bike_accessible = false AND rj.location IS NOT NULL
                       RETURN rj, point.distance(osmLoc, rj.location) AS dist
                       ORDER BY dist ASC
                       LIMIT 1
                     }
                     MERGE (osmn)-[r:NEAR_BIKE]->(rj)
                       ON CREATE SET r.distance = dist, r.status = 'nearest_non_bike_accessible'",
                    {batchSize:500, iterateList:true, parallel:false}
                )
                YIELD batches, total, errorMessages
                RETURN batches, total, errorMessages
            """).single()
            print(f"  NEAR_BIKE step2: batches={r['batches']}, total={r['total']}, errors={r['errorMessages']}")

            # Step 3: Way POIs — at least 1 node connected to bike_accessible
            r = session.run("""
                CALL apoc.periodic.iterate(
                    "MATCH (osmn:OSMNode)-[:NEAR_BIKE]->(nonRJ:RoadJunction)
                     WHERE nonRJ.bike_accessible = false
                       AND NOT (osmn)-[:NEAR_BIKE]->(:RoadJunction {bike_accessible: true})
                       AND osmn.lat IS NOT NULL AND osmn.lon IS NOT NULL
                     RETURN osmn",
                    "WITH osmn, point({latitude: toFloat(osmn.lat), longitude: toFloat(osmn.lon)}) AS osmLoc
                     CALL {
                       WITH osmn, osmLoc
                       MATCH (bikeRJ:RoadJunction)
                       WHERE bikeRJ.bike_accessible = true AND bikeRJ.location IS NOT NULL
                       RETURN bikeRJ, point.distance(osmLoc, bikeRJ.location) AS dist
                       ORDER BY dist ASC
                       LIMIT 1
                     }
                     MERGE (osmn)-[r:NEAR_BIKE]->(bikeRJ)
                       ON CREATE SET r.distance = dist, r.status = 'nearest_bike_from_non_bike'",
                    {batchSize:500, iterateList:true, parallel:false}
                )
                YIELD batches, total, errorMessages
                RETURN batches, total, errorMessages
            """).single()
            print(f"  NEAR_BIKE step3: batches={r['batches']}, total={r['total']}, errors={r['errorMessages']}")

            # --- final count ---
            car_count = session.run("MATCH ()-[r:NEAR_CAR]->() RETURN count(r) AS c").single()['c']
            bike_count = session.run("MATCH ()-[r:NEAR_BIKE]->() RETURN count(r) AS c").single()['c']
            print(f"  [result] NEAR_CAR: {car_count}, NEAR_BIKE: {bike_count}")


def add_options():
    parser = argparse.ArgumentParser(description='Insertion of POI in the graph.')
    parser.add_argument('--neo4jURL', '-n', dest='neo4jURL', type=str, required=True)
    parser.add_argument('--neo4juser', '-u', dest='neo4juser', type=str, required=True)
    parser.add_argument('--neo4jpwd', '-p', dest='neo4jpwd', type=str, required=True)
    parser.add_argument('--latitude', '-x', dest='lat', type=float, required=True)
    parser.add_argument('--longitude', '-y', dest='lon', type=float, required=True)
    parser.add_argument('--distance', '-d', dest='dist', type=float, required=True)
    parser.add_argument('--spatial', '-s', dest='spatial', type=str, required=False, default='False')
    return parser


def main(args=None):
    start_time = time.time()
    argParser = add_options()
    options = argParser.parse_args(args=args)

    dist = options.dist
    lon = options.lon
    lat = options.lat
    greeter = App(options.neo4jURL, options.neo4juser, options.neo4jpwd)
    neo4j_home = greeter.get_path()[0][0]
    import_dir = greeter.get_import_folder_name()[0][0]
    if os.path.isabs(import_dir):
        path = import_dir + "\\"
    else:
        path = neo4j_home + '\\' + import_dir + "\\"

    # ==================================================================
    # QUERY 1: way POIs + their member geometry nodes
    # [CHANGE 1] expanded: amenity + shop + railway + public_transport
    # ==================================================================
    print("Query 1: Downloading way POIs...")
    query1 = (
        f'[out:json][timeout:120];'
        f'(way(around:{dist},{lat},{lon})["amenity"];'
        f'way(around:{dist},{lat},{lon})["shop"];'
        f'way(around:{dist},{lat},{lon})["railway"="station"];'
        f'way(around:{dist},{lat},{lon})["public_transport"="station"];'
        f');(._;>;);out body;'
    )
    result = overpass_query(query1)

    list_node_way = []
    for w in result.ways:
        for n in w.get_nodes(resolve_missing=False):
            d = {'type': 'node', 'id': n.id,
                 'id_way': w.id,
                 'lat': str(n.lat),
                 'lon': str(n.lon),
                 'geometry': 'POINT(' + str(n.lat) + ' ' + str(n.lon) + ')',
                 'tags': n.tags}
            list_node_way.append(d)
    res = {"elements": list_node_way}
    with open(path + 'nodeway.json', "w") as f:
        json.dump(res, f)
    print(f"  nodeway.json generated ({len(list_node_way)} nodes)")

    greeter.import_node_way()

    # ==================================================================
    # QUERY 2: way POIs with tags (for import_way)
    # Uses same result from query 1 — no extra Overpass call needed
    # ==================================================================
    list_way = []
    for way in result.ways:
        d = {'type': 'way', 'id': way.id, 'tags': way.tags}
        d['nodes'] = [node.id for node in way.nodes]
        list_way.append(d)
    res = {"elements": list_way}
    with open(path + "wayfile.json", "w") as f:
        json.dump(res, f)
    print(f"  wayfile.json generated ({len(list_way)} ways)")

    greeter.import_way()
    print("  import wayfile.json: done")

    # ==================================================================
    # QUERY 3: node POIs (single-node amenities, shops, stops, stations)
    # [CHANGE 1] expanded categories
    # [CHANGE 6] removed duplicate import_way() call
    # ==================================================================
    print("Query 3: Downloading node POIs...")
    time.sleep(15)
    query3 = (
        f'[out:json][timeout:120];'
        f'(node["amenity"](around:{dist},{lat},{lon});'
        f'node["shop"](around:{dist},{lat},{lon});'
        f'node["public_transport"="platform"](around:{dist},{lat},{lon});'
        f'node["public_transport"="stop_position"](around:{dist},{lat},{lon});'
        f'node["highway"="bus_stop"](around:{dist},{lat},{lon});'
        f'node["railway"="station"](around:{dist},{lat},{lon});'
        f');out body;'
    )
    result = overpass_query(query3)

    list_node = []
    for node in result.nodes:
        d = {'type': 'node', 'id': node.id,
             'lat': str(node.lat),
             'lon': str(node.lon),
             'geometry': 'POINT(' + str(node.lat) + ' ' + str(node.lon) + ')',
             'tags': node.tags}
        list_node.append(d)
    res = {"elements": list_node}
    with open(path + 'nodefile.json', "w") as f:
        json.dump(res, f)
    print(f"  nodefile.json generated ({len(list_node)} nodes)")

    greeter.import_node()
    print("  import nodefile.json: done")

    # ==================================================================
    # POST-IMPORT: spatial, location, indexing, marking, connecting
    # ==================================================================
    if options.spatial == 'True':
        print("Adding nodes to spatial layer...")
        greeter.import_nodes_into_spatial_layer()

    print("Setting locations...")
    greeter.set_location()

    print("Creating indexes...")
    greeter.set_index()

    print("Marking accessible junctions (driveable + bike_accessible)...")
    greeter.mark_accessible_roadjunctions()

    print("Connecting OSMNodes to nearest junctions (NEAR_CAR + NEAR_BIKE within 100m)...")
    greeter.connect_amenity()

    greeter.close()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    return 0


main()
