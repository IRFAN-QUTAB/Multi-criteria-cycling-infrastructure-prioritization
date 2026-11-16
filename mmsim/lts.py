import ast
import re

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
LANE_DEFAULTS = {
    'motorway': 2, 'trunk': 2, 'primary': 2, 'secondary': 2,
    'tertiary': 2, 'unclassified': 2, 'residential': 2,
    'living_street': 1, 'service': 1, 'pedestrian': 1,
    'cycleway': 1, 'path': 1, 'footway': 1, 'track': 1,
}
SPEED_STRINGS = {
    'it:urban': 50, 'it:rural': 90, 'it:motorway': 130,
    'it:living_street': 20, 'it:zone30': 30, 'it:zone:30': 30,
    'walk': 10, 'none': 130,
}
CLASS2_ORIG = {'lane', 'shared_lane', 'opposite_lane', 'share_busway',
               'shared'}
CLASS2_MAIN = {'lane', 'opposite_lane', 'share_busway'}
ZONE30_TYPES = {'residential', 'unclassified'}


def first(value):
    """First usable value; also parses list strings written by GraphML."""
    if isinstance(value, list):
        return value[0] if value else None
    if isinstance(value, str):
        s = value.strip()
        if s.startswith('[') and s.endswith(']'):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return v[0] if v else None
            except (ValueError, SyntaxError):
                pass
        return s
    return value


def as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ('true', '1', 'yes')


def speed_is_tagged(maxspeed):
    v = first(maxspeed)
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in SPEED_STRINGS or bool(re.findall(r'\d+', s))


def parse_speed(value, highway, zone30=False):
    value = first(value)
    hw = first(highway)
    default = SPEED_DEFAULTS.get(hw, 50)
    if zone30 and hw in ZONE30_TYPES:
        default = 30
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in SPEED_STRINGS:
        return SPEED_STRINGS[s]
    is_mph = 'mph' in s
    nums = re.findall(r'\d+', s)
    if not nums:
        return default
    speed = int(nums[0])
    if is_mph:
        speed = int(round(speed * 1.60934))
    return speed


def parse_lanes(value, highway):
    value = first(value)
    hw = first(highway)
    if value is not None:
        nums = re.findall(r'\d+', str(value))
        if nums:
            return int(nums[0])
    return LANE_DEFAULTS.get(hw, 2)


def bicycle_class(p, class2):
    highway = first(p.get('highway'))
    bicycle = first(p.get('bicycle'))
    segregated = first(p.get('segregated'))
    cw = {first(p.get('cycleway')), first(p.get('cycleway:left')),
          first(p.get('cycleway:right')), first(p.get('cycleway:both'))}
    cw.discard(None)
    if highway == 'cycleway':
        return 1
    if highway == 'path' and bicycle == 'designated':
        return 1
    if highway in ('path', 'footway') and segregated == 'yes':
        return 1
    if any(v in {'track', 'opposite_track'} for v in cw):
        return 1
    if any(v in class2 for v in cw):
        return 2
    return 3


def lts_value(bclass, speed, lanes, highway, on_bus):
    hw = first(highway)
    if bclass == 1:
        return 1
    if bclass == 2:
        if speed <= 50 and lanes <= 2:
            return 1
        if (speed <= 50 and lanes <= 4) or (speed < 50 and on_bus):
            return 2
        if (speed <= 60 and lanes <= 4) or (speed < 70 and on_bus):
            return 3
        return 4
    if hw in ('trunk', 'trunk_link'):
        return 4
    if hw in ('primary', 'primary_link'):
        if speed > 50 or lanes > 2:
            return 4
        return 3
    if hw in ('secondary', 'secondary_link'):
        if speed > 50 or lanes > 3:
            return 4
        return 3
    if hw in ('tertiary', 'tertiary_link'):
        if speed > 50 or lanes > 3:
            return 3
        if on_bus:
            return 3
        return 2
    if (speed <= 30 and lanes <= 2) or (speed <= 20 and lanes <= 3):
        return 1
    if speed <= 50 and lanes <= 3:
        if on_bus:
            return 3
        return 2
    if speed <= 50 and lanes <= 4:
        return 3
    return 4


def edge_lts(p, variant):
    """LTS of one ROUTE edge (properties dict) or None if not bikeable."""
    if not as_bool(p.get('bike_allowed')):
        return None
    if variant == "stored":
        try:
            return int(float(p.get('lts')))
        except (TypeError, ValueError):
            return None
    class2 = CLASS2_ORIG if variant == "orig" else CLASS2_MAIN
    zone30 = variant == "zone30"
    hw = p.get('highway')
    speed = parse_speed(p.get('maxspeed'), hw, zone30=zone30)
    lanes = parse_lanes(p.get('lanes'), hw)
    on_bus = as_bool(p.get('on_bus_route'))
    return lts_value(bicycle_class(p, class2), speed, lanes, hw, on_bus)
