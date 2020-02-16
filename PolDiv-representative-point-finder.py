import json
import statistics
import time
from collections import defaultdict, namedtuple
from math import sqrt

import pandas as pd
import shapefile
from shapely.geometry import Polygon, Point
from tqdm import tqdm

from toolbox import generic_esri_reader


def PolDiv_representative_point_finder(prov_id=35):
    pol_divs = "./geo_data/polling_divisions_boundaries_2015_shp/poll_div_bounds_2015.shp"
    diss_block = "./geo_data/dissemination_blocks_cartographic/ldb_000b16a_e.shp"
    diss_block_data = "./geo_data/DB.csv"
    pd_db_association = "./geo_data/PolDiv_DB_association_prov_35_20200215_215127.json"
    print("Reading files")
    pol_divs = generic_esri_reader(pol_divs)
    diss_block = generic_esri_reader(diss_block)
    diss_block_data = pd.read_csv(diss_block_data)
    pd_db_association = json.load(open(pd_db_association))
    print("    Done reading data")
    print(f"Filtering by province id {prov_id}")
    pol_divs = [pd for pd in pol_divs if prov_id * 1000 <= pd.get("fed_num") < (prov_id + 1) * 1000]
    diss_block = [db for db in diss_block if db.get("pruid") == str(prov_id) or db.get("pruid") == prov_id]
    diss_block_data = diss_block_data[diss_block_data["DBuid"] < 1000000000 * (prov_id + 1)]
    diss_block_data = diss_block_data[diss_block_data["DBuid"] >= 1000000000 * prov_id]
    print(f"    Done filtering by province id {prov_id}")
    results = find_representative_points(pd_db_association, diss_block_data, pol_divs, diss_block)
    with open(f"representative_point_{prov_id}.json", "w", encoding="utf-8") as jsonfile:
        json.dump(results, jsonfile, indent=4)
    print("Fin")


def find_representative_points(pd_db_association, diss_block_data, pol_divs, diss_block_geospatial):
    diss_block_geospatial = {db.get("dbuid"): db for db in diss_block_geospatial}
    results = defaultdict(dict)
    for pol_div in tqdm(pol_divs, desc="Finding Map Centers"):
        results[get_pol_div_str(pol_div)]["MapCenter"] = list(list(build_shape(pol_div).centroid.coords).pop())
    Location = namedtuple("Location", ["x", "y", "pop"])
    for pol_div, dis_blocks in tqdm(pd_db_association.items(), desc="Finding Other Centers"):
        db_pops = []
        for dis_block in dis_blocks:
            data = diss_block_data[diss_block_data["DBuid"] == int(dis_block)].iloc[0].to_dict()
            try:
                pop = int(data.get("DBpop_2016", 0))
            except ValueError:
                pop = 0
            x = diss_block_geospatial[dis_block]['dbrplamx']
            y = diss_block_geospatial[dis_block]['dbrplamy']
            db_pops.append(Location(x, y, pop))
        results[pol_div]["MeanCenter"] = mean_center(db_pops)
        results[pol_div]["MedianCenter"] = median_center(db_pops)
        results[pol_div]["GeometricCenter"] = geometric_center(db_pops)
    return results


def mean_center(db_pops):
    try:
        x, y, pop = zip(*db_pops)
        _x = sum(i * j for i, j in zip(x, pop)) / sum(pop)
        _y = sum(i * j for i, j in zip(y, pop)) / sum(pop)
        center = (_x, _y)
        return center
    except:
        return None


def median_center(db_pops):
    try:
        x, y = [], []
        for DB in db_pops:
            for _ in range(int(DB.pop)):
                x.append(DB.x)
                y.append(DB.y)
        center = (statistics.median(x), statistics.median(y))
        return center
    except:
        return None


def geometric_center(db_pops, init_center=None, max_time=120, epsilon=0.1, max_iter=1000):
    try:
        if init_center is None:
            init_center = mean_center(db_pops)
        elements = [(DB.x, DB.y) for DB in db_pops for _ in range(DB.pop)]
        y = [init_center]
        t0, i = time.time(), 0
        while (time.time() - t0) < max_time and i < max_iter:
            i += 1
            numerator = __compute_numerator(x=elements, y_i=y[-1])
            denominator = __compute_denominator(x=elements, y_i=y[-1])
            if denominator == 0:
                break
            _y = tuple(e / denominator for e in numerator)
            y.append(_y)
            if euclidean_dist(y[-1], y[-2]) < epsilon:
                break
        return y[-1]
    except:
        return None


def __compute_numerator(x, y_i):
    _x = []
    for x_j in x:
        _x.append(tuple(e / euclidean_dist(x_j, y_i) for e in x_j))
    i, j = zip(*_x)
    _x = (sum(i), sum(j))
    return _x


def __compute_denominator(x, y_i):
    return sum((1 / euclidean_dist(x_j, y_i)) for x_j in x)


def euclidean_dist(p1, p2):
    if isinstance(p1, Point):
        p1 = (p1.x, p1.y)
    if isinstance(p2, Point):
        p2 = (p2.x, p2.y)
    return sqrt(sum(pow(a - b, 2) for a, b in zip(p1, p2)))


def get_pol_div_str(PolDiv):
    if PolDiv.get('pd_type') == 'M':
        return f"{PolDiv.get('fed_num')}-{PolDiv.get('pd_num')}-{PolDiv.get('pd_nbr_sfx', 0)}-" \
               f"{PolDiv.get('poll_name').replace('/', '').replace('-', '').replace(' ', '')}-" \
               f"{PolDiv.get('bldg_namee').replace('-', '').replace(' ', '')}"
    return f"{PolDiv.get('fed_num')}-{PolDiv.get('pd_num')}-{PolDiv.get('pd_nbr_sfx', 0)}"


def build_shape(shape, allow_holes=True):
    if not isinstance(shape, shapefile.Shape):
        shape = shape["shape"]
    parts = list(shape.parts) + [-1]
    holes = [Polygon(list(shape.points)[i:j]) for i, j in zip(parts, parts[1:])]
    polygon = holes.pop(0)
    if allow_holes:
        holes = holes if len(holes) > 0 else None
    else:
        holes = None
    polygon = Polygon(polygon, holes=holes)
    while not polygon.is_valid:
        polygon = polygon.buffer(-1)
    return polygon


if __name__ == '__main__':
    PolDiv_representative_point_finder()
