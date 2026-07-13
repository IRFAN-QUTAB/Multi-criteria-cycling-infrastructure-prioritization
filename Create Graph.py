import osmnx as ox
import argparse
from neo4j import GraphDatabase
import os
import time
import re
import requests

# ============================================================================
#  CLASSIFICATION HELPERS  (bicycle class + LTS, based on Vierø & Szell 2025,
# ============================================================================

# Italian default speed limits (km/h) used when OSM has no maxspeed tag.
SPEED_DEFAULTS = {
    'motorway': 130, 'motorway_link': 80,
    'trunk': 90, 'trunk_link': 70,
    'primary': 50, 'primary_link': 50,
    'secondary': 50, 'secondary_link': 50,
    'tertiary': 50, 'tertiary_link': 50,
    'unclassified': 50,
    'residential': 50,
    'living_street': 20,
    'service': 30,
    'pedestrian': 10,
    'track': 30,
    'cycleway': 30, 'path': 20, 'footway': 10,
}

# Default lane counts used when OSM has no lanes tag.
LANE_DEFAULTS = {
    'motorway': 2, 'trunk': 2, 'primary': 2, 'secondary': 2,
    'tertiary': 2, 'unclassified': 2, 'residential': 2,
    'living_street': 1, 'service': 1, 'pedestrian': 1,
    'cycleway': 1, 'path': 1, 'footway': 1, 'track': 1,
}

# [CHANGE 5] trunk/trunk_link REMOVED from forbidden list.
# Now only motorway is auto-forbidden for bikes.  Trunk roads are
# allowed at LTS 4 unless bicycle=no is explicitly set.
BIKE_FORBIDDEN_HIGHWAYS = {
    'motorway', 'motorway_link', 'steps',
}

# Highway types a car cannot use.
# [CHANGE 2] 'service' REMOVED — service roads are driveable by default.
# [CHANGE 3] 'track' REMOVED — handled conditionally via tracktype/surface.
NON_CAR_HIGHWAYS = {
    'cycleway', 'footway', 'path', 'pedestrian', 'steps',
    'bridleway', 'corridor',
}

# Surfaces unsuitable for everyday cycling (Vierø & Szell exclude these).
BAD_SURFACES = {'grass', 'sand', 'dirt', 'mud', 'ground', 'earth', 'clay'}

# Surfaces considered paved/firm enough for car driving on track roads.
DRIVEABLE_SURFACES = {
    'asphalt', 'paved', 'concrete', 'concrete:plates', 'concrete:lanes',
    'paving_stones', 'sett', 'cobblestone', 'compacted',
}

# Italian special maxspeed string values.
SPEED_STRINGS = {
    'it:urban': 50, 'it:rural': 90, 'it:motorway': 130,
    'it:living_street': 20, 'it:zone30': 30, 'it:zone:30': 30,
    'walk': 10, 'none': 130,
}


def _first(value):
    """OSM tags sometimes arrive as a list; return the first usable value."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def parse_speed(value, highway):
    """Return an integer speed limit (km/h), imputing from road type if missing."""
    value = _first(value)
    hw = _first(highway)
    if value is None:
        return SPEED_DEFAULTS.get(hw, 50)
    s = str(value).strip().lower()
    if s in SPEED_STRINGS:
        return SPEED_STRINGS[s]
    is_mph = 'mph' in s
    nums = re.findall(r'\d+', s)
    if not nums:
        return SPEED_DEFAULTS.get(hw, 50)
    speed = int(nums[0])
    if is_mph:
        speed = int(round(speed * 1.60934))
    return speed


def parse_lanes(value, highway):
    """Return an integer lane count, imputing from road type if missing."""
    value = _first(value)
    hw = _first(highway)
    if value is not None:
        nums = re.findall(r'\d+', str(value))
        if nums:
            return int(nums[0])
    return LANE_DEFAULTS.get(hw, 2)


def get_bicycle_class(data):
    """
    Class 1 = protected/separated track  -> always LTS 1
    Class 2 = painted lane (no physical separation)
    Class 3 = mixed traffic, no infrastructure

    [CHANGE 4] Added:
      - segregated=yes on shared paths -> Class 1
      - cycleway=shared -> Class 2
    """
    highway = _first(data.get('highway'))
    bicycle = _first(data.get('bicycle'))
    segregated = _first(data.get('segregated'))
    cw_values = {
        _first(data.get('cycleway')),
        _first(data.get('cycleway:left')),
        _first(data.get('cycleway:right')),
        _first(data.get('cycleway:both')),
    }
    cw_values.discard(None)

    # Class 1: separated infrastructure
    if highway == 'cycleway':
        return 1
    if highway == 'path' and bicycle == 'designated':
        return 1
    if highway in ('path', 'footway') and segregated == 'yes':
        return 1
    if any(v in {'track', 'opposite_track'} for v in cw_values):
        return 1

    # Class 2: painted / on-road lane or shared marking
    if any(v in {'lane', 'shared_lane', 'opposite_lane', 'share_busway', 'shared'} for v in cw_values):
        return 2

    # Class 3: mixed traffic
    return 3


def get_lts(bicycle_class, speed, lanes, highway, on_bus_route):
    """Cascade from low stress to high, per Vierø & Szell 2025, Table 1."""
    hw = _first(highway)

    if bicycle_class == 1:
        return 1

    if bicycle_class == 2:
        if speed <= 50 and lanes <= 2:
            return 1
        if (speed <= 50 and lanes <= 4) or (speed < 50 and on_bus_route):
            return 2
        if (speed <= 60 and lanes <= 4) or (speed < 70 and on_bus_route):
            return 3
        return 4

    # bicycle_class == 3 (mixed traffic)
    # Major roads with no bike infra are inherently high-stress
    # regardless of speed/lanes — highway type sets minimum LTS.

    # Trunk/trunk_link with mixed traffic → always LTS 4
    if hw in ('trunk', 'trunk_link'):
        return 4

    # Primary with mixed traffic → LTS 3 minimum
    if hw in ('primary', 'primary_link'):
        if speed > 50 or lanes > 2:
            return 4
        return 3

    # Secondary with mixed traffic → LTS 3 minimum
    if hw in ('secondary', 'secondary_link'):
        if speed > 50 or lanes > 3:
            return 4
        return 3

    # Tertiary with mixed traffic → LTS 2 minimum
    if hw in ('tertiary', 'tertiary_link'):
        if speed > 50 or lanes > 3:
            return 3
        if on_bus_route:
            return 3
        return 2

    # Remaining minor roads (residential, unclassified, service, etc.)
    if (speed <= 30 and lanes <= 2) or (speed <= 20 and lanes <= 3):
        return 1
    if speed <= 50 and lanes <= 3:
        if on_bus_route:
            return 3
        return 2
    if speed <= 50 and lanes <= 4:
        return 3
    return 4


def is_driveable(data):
    """
    Can a car legally use this edge?

    [CHANGE 2] service roads are now driveable unless access=private/no.
    [CHANGE 3] track roads are conditionally driveable based on
               tracktype (grade1/grade2) or surface (paved/asphalt/etc).
    """
    highway = _first(data.get('highway'))
    access = _first(data.get('access'))
    motor = _first(data.get('motor_vehicle'))

    # Always non-driveable highway types
    if highway in NON_CAR_HIGHWAYS:
        return False

    # Access restrictions
    if access in {'no', 'private'}:
        return False
    if motor in {'no', 'private'}:
        return False

    # [CHANGE 2] service roads: only exclude if explicitly private
    if highway == 'service':
        service_type = _first(data.get('service'))
        if service_type in {'driveway', 'parking_aisle'}:
            return True
        if access in {'no', 'private'}:
            return False
        return True

    # [CHANGE 3] track roads: check tracktype and surface
    if highway == 'track':
        tracktype = _first(data.get('tracktype'))
        surface = _first(data.get('surface'))
        if tracktype in {'grade1', 'grade2'}:
            return True
        if surface in DRIVEABLE_SURFACES:
            return True
        return False

    return True


def is_bike_allowed(data):
    """
    Can a bicycle legally and practically use this edge?

    [CHANGE 4] bicycle=use_sidepath -> False (cyclist must use separate path).
    [CHANGE 5] trunk/trunk_link now allowed unless bicycle=no explicitly.
    """
    highway = _first(data.get('highway'))
    access = _first(data.get('access'))
    bicycle = _first(data.get('bicycle'))
    surface = _first(data.get('surface'))

    # Explicit denial
    if bicycle in {'no', 'dismount', 'private', 'use_sidepath'}:
        return False
    if access in {'no', 'private'} and bicycle not in {'yes', 'designated', 'permissive'}:
        return False

    # Auto-forbidden highways (only motorway now; trunk removed)
    if highway in BIKE_FORBIDDEN_HIGHWAYS and bicycle not in {'yes', 'designated'}:
        return False

    # Footway/pedestrian: only if bike explicitly allowed
    if highway in {'footway', 'pedestrian'} and bicycle not in {'yes', 'designated', 'permissive'}:
        return False

    # Bad surface
    if surface in BAD_SURFACES:
        return False

    return True


def get_route_type(driveable, bike_allowed):
    """
    [CHANGE 6] Classify each edge into one of four route types.

    car_only   -> only cars can use this road
    bike_only  -> only bikes (cycleway, path, etc.)
    shared     -> both car and bike can use this road
    restricted -> neither car nor bike (e.g. steps, private, bad surface)
    """
    if driveable and bike_allowed:
        return 'shared'
    if driveable and not bike_allowed:
        return 'car_only'
    if not driveable and bike_allowed:
        return 'bike_only'
    return 'restricted'


# ============================================================================
#  BUS ROUTE FETCH  (route=bus relations are NOT returned by OSMnx graph
#  download, so we query Overpass separately for their member way ids)
# ============================================================================

def fetch_bus_route_ways(south, west, north, east):
    """Return a set of OSM way ids that belong to any route=bus relation."""
    query = (
        f'[out:json][timeout:180];'
        f'rel["route"="bus"]({south},{west},{north},{east});'
        f'way(r);'
        f'out ids;'
    )
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'UrbanNetworkAnalysis/1.0',
    }
    try:
        resp = requests.get(
            "https://overpass-api.de/api/interpreter",
            params={'data': query},
            headers=headers,
            timeout=200,
        )
        resp.raise_for_status()
        elements = resp.json().get('elements', [])
        ids = {el['id'] for el in elements if el.get('type') == 'way'}
        print(f"Bus route ways found: {len(ids)}")
        return ids
    except Exception as e:
        print(f"WARNING: bus route fetch failed ({e}). Continuing with no bus routes.")
        return set()


# ============================================================================
#  NEO4J APP  (original class, extended with new conversion method)
# ============================================================================

class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(
            uri, auth=(user, password),
            notifications_min_severity="OFF"
        )

    def close(self):
        self.driver.close()

    def creation_graph(self, file):
        with self.driver.session() as session:
            result = session.execute_write(self._creation_graph, file)
            return result

    @staticmethod
    def _creation_graph(tx, file):
        result = tx.run("""
                        CALL apoc.import.graphml($file, {storeNodeIds: true, defaultRelationshipType: 'ROUTE'});
                    """, file=file)
        return result.values()

    def get_path(self):
        with self.driver.session() as session:
            result = session.execute_write(self._get_path)
            return result

    @staticmethod
    def _get_path(tx):
        # Neo4j 5.x renamed config keys: dbms.directories.* -> server.directories.*
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

    def set_label(self):
        with self.driver.session() as session:
            result = session.execute_write(self._creation_label)
            return result

    @staticmethod
    def _creation_label(tx):
        result = tx.run("""
                        MATCH (n) SET n:RoadJunction;
                    """)
        return result.values()

    def set_location(self):
        with self.driver.session() as session:
            result = session.execute_write(self._creation_location)
            return result

    @staticmethod
    def _creation_location(tx):
        result = tx.run("""
                           MATCH (n:RoadJunction) SET n.location = point({latitude: tofloat(n.y), longitude: tofloat(n.x)}),
                                            n.lat = tofloat(n.y), 
                                            n.lon = tofloat(n.x),
                                            n.geometry='POINT(' + n.y + ' ' + n.x +')';
                       """)
        return result.values()

    def set_distance(self):
        with self.driver.session() as session:
            result = session.execute_write(self._set_distance)
            return result

    @staticmethod
    def _set_distance(tx):
        result = tx.run("""
                           MATCH (n:RoadJunction)-[r:ROUTE]-() SET r.distance=tofloat(r.length), r.status='active'
                       """)
        return result.values()

    def set_bike_attributes(self):
        """Type-convert the bicycle / LTS / route_type attributes from strings."""
        with self.driver.session() as session:
            result = session.execute_write(self._set_bike_attributes)
            return result

    @staticmethod
    def _set_bike_attributes(tx):
        result = tx.run("""
            MATCH ()-[r:ROUTE]->()
            SET r.speed         = CASE WHEN r.speed IS NOT NULL THEN toInteger(r.speed) ELSE null END,
                r.lanes         = CASE WHEN r.lanes IS NOT NULL THEN toInteger(r.lanes) ELSE null END,
                r.bicycle_class = CASE WHEN r.bicycle_class IS NOT NULL AND r.bicycle_class <> 'None'
                                       THEN toInteger(r.bicycle_class) ELSE null END,
                r.lts           = CASE WHEN r.lts IS NOT NULL AND r.lts <> 'None'
                                       THEN toInteger(r.lts) ELSE null END,
                r.driveable     = CASE WHEN toLower(toString(r.driveable)) = 'true' THEN true ELSE false END,
                r.bike_allowed  = CASE WHEN toLower(toString(r.bike_allowed)) = 'true' THEN true ELSE false END,
                r.on_bus_route  = CASE WHEN toLower(toString(r.on_bus_route)) = 'true' THEN true ELSE false END
        """)
        return result.values()

    def rename_junction_id(self):
        with self.driver.session() as session:
            return session.execute_write(self._rename_junction_id)

    @staticmethod
    def _rename_junction_id(tx):
        return tx.run("""
            MATCH (n:RoadJunction)
            WHERE n.id IS NOT NULL
            SET n.osmnode_id = n.id
            REMOVE n.id
        """).values()

    def rename_route_osmid(self):
        with self.driver.session() as session:
            return session.execute_write(self._rename_route_osmid)

    @staticmethod
    def _rename_route_osmid(tx):
        return tx.run("""
            MATCH ()-[r:ROUTE]->()
            WHERE r.osmid IS NOT NULL
            SET r.osmway_id = r.osmid
            REMOVE r.osmid
        """).values()

    def set_index(self):
        with self.driver.session() as session:
            result = session.execute_write(self._set_index)
            return result

    @staticmethod
    def _set_index(tx):
        tx.run("CREATE INDEX IF NOT EXISTS FOR (n:RoadJunction) ON (n.osmnode_id)")
        tx.run("CREATE INDEX IF NOT EXISTS FOR (n:RoadJunction) ON (n.location)")
        tx.run("CREATE INDEX IF NOT EXISTS FOR (n:RoadJunction) ON (n.lat)")
        tx.run("CREATE INDEX IF NOT EXISTS FOR (n:RoadOsm) ON (n.name)")
        tx.run("CREATE INDEX IF NOT EXISTS FOR (n:POI) ON (n.osm_id)")
        print("  Indexes created: RoadJunction(osmnode_id, location, lat), RoadOsm(name), POI(osm_id)")
        return []

    def generate_spatial_layer(self):
        """Create spatial layer if neo4j-spatial plugin is installed."""
        try:
            with self.driver.session() as session:
                result = session.execute_write(self._generate_spatial_layer)
                return result
        except Exception as e:
            print(f"  Spatial layer skipped (plugin not installed): {e}")
            return []

    @staticmethod
    def _generate_spatial_layer(tx):
        result = tx.run("CALL spatial.layers()")
        if len(result.values()) == 0:
            result = tx.run("CALL spatial.addWKTLayer('spatial', 'geometry')")
        return result.values()

    def import_nodes_in_spatial_layer(self):
        """Add nodes to spatial layer if plugin is installed."""
        try:
            with self.driver.session() as session:
                result = session.execute_write(self._import_nodes_in_spatial_layer)
                return result
        except Exception as e:
            print(f"  Spatial import skipped (plugin not installed): {e}")
            return []

    @staticmethod
    def _import_nodes_in_spatial_layer(tx):
        result = tx.run("""
        MATCH (n:RoadJunction)
        CALL spatial.addNode('spatial', n) YIELD node RETURN node
        """)
        return result.values()


def add_options():
    parser = argparse.ArgumentParser(description='Creation of unified road + cycle routing graph.')
    parser.add_argument('--latitude', '-x', dest='lat', type=float, required=True)
    parser.add_argument('--longitude', '-y', dest='lon', type=float, required=True)
    parser.add_argument('--distance', '-d', dest='dist', type=float, required=True)
    parser.add_argument('--neo4jURL', '-n', dest='neo4jURL', type=str, required=True)
    parser.add_argument('--neo4juser', '-u', dest='neo4juser', type=str, required=True)
    parser.add_argument('--neo4jpwd', '-p', dest='neo4jpwd', type=str, required=True)
    parser.add_argument('--nameFile', '-f', dest='file_name', type=str, required=True)
    return parser


def main(args=None):
    start_time = time.time()
    argParser = add_options()
    options = argParser.parse_args(args=args)
    greeter = App(options.neo4jURL, options.neo4juser, options.neo4jpwd)
    neo4j_home = greeter.get_path()[0][0]
    import_dir = greeter.get_import_folder_name()[0][0]
    # Neo4j 5.x returns absolute path for import dir; older versions return relative
    if os.path.isabs(import_dir):
        path = os.path.join(import_dir, options.file_name)
    else:
        path = os.path.join(neo4j_home, import_dir, options.file_name)

    # --- make OSMnx keep the bicycle-relevant way tags as edge attributes ---
    extra_tags = [
        'cycleway', 'cycleway:left', 'cycleway:right', 'cycleway:both',
        'bicycle', 'surface', 'segregated', 'foot', 'motor_vehicle', 'psv',
        'service', 'tracktype',
    ]
    ox.settings.useful_tags_way = list(set(ox.settings.useful_tags_way + extra_tags))

    # --- download the FULL network (roads + cycleways + paths), unsimplified ---
    # Explicit filter avoids garbage highway types that network_type='all'
    # can include (construction, proposed, abandoned, platform, raceway,
    # corridor, elevator, bus_guideway, rest_area).
    # Excluded: steps (no car/bike use), bridleway (irrelevant for Italy).
    # Kept: footway (rare bicycle=yes cases), pedestrian (Italian ZTL zones).
    HIGHWAY_FILTER = (
        '["highway"~"motorway|motorway_link|trunk|trunk_link'
        '|primary|primary_link|secondary|secondary_link'
        '|tertiary|tertiary_link|unclassified|residential'
        '|living_street|service|pedestrian|track'
        '|cycleway|path|footway"]'
    )
    G = ox.graph_from_point(
        (options.lat, options.lon),
        dist=int(options.dist),
        dist_type='bbox',
        simplify=False,
        custom_filter=HIGHWAY_FILTER
    )

    # --- bounding box from the downloaded nodes, then fetch bus-route way ids ---
    ys = [d['y'] for _, d in G.nodes(data=True)]
    xs = [d['x'] for _, d in G.nodes(data=True)]
    south, north = min(ys), max(ys)
    west, east = min(xs), max(xs)
    bus_way_ids = fetch_bus_route_ways(south, west, north, east)

    # --- classify every edge ---
    for u, v, k, data in G.edges(keys=True, data=True):
        highway = data.get('highway')

        # is this edge on a bus route?
        osmid = data.get('osmid')
        id_list = osmid if isinstance(osmid, list) else [osmid]
        on_bus = any((i is not None) and (int(i) in bus_way_ids) for i in id_list)

        speed = parse_speed(data.get('maxspeed'), highway)
        lanes = parse_lanes(data.get('lanes'), highway)
        bclass = get_bicycle_class(data)

        drv = is_driveable(data)
        bike = is_bike_allowed(data)

        data['driveable'] = drv
        data['bike_allowed'] = bike
        data['on_bus_route'] = on_bus
        data['speed'] = speed
        data['lanes'] = lanes

        # [CHANGE 1] LTS only for bicycle-accessible edges
        if bike:
            data['bicycle_class'] = bclass
            data['lts'] = get_lts(bclass, speed, lanes, highway, on_bus)
        else:
            data['bicycle_class'] = None
            data['lts'] = None

        # [CHANGE 6] route type classification
        data['route_type'] = get_route_type(drv, bike)

    ox.save_graphml(G, path)

    greeter.generate_spatial_layer()          # set up spatial layer first
    greeter.creation_graph(options.file_name)  # create the graph from .graphml file
    greeter.set_label()                       # set labels on nodes
    greeter.rename_junction_id()
    greeter.set_location()                    # set location attributes for nodes
    greeter.set_distance()                    # set distance + status on relationships
    greeter.set_bike_attributes()             # type-convert lts / class / flags
    greeter.rename_route_osmid()
    greeter.set_index()                       # set index for optimized queries
    greeter.close()                           # close the connection last

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Execution time: {elapsed:.2f} seconds")
    return 0


main()
