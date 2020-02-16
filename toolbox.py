import datetime
import os

import shapefile
from tqdm import tqdm


def time_for_filename(time=datetime.datetime.now()):
    """
    Gets a time string for file names that I like.
    """
    time = time.strftime("%Y%m%d_%H%M%S")
    return time


def generic_esri_reader(shape_file, encoding='ISO-8859-1'):
    """Reads and parses the shape file into a more usable format"""
    file = []
    with shapefile.Reader(shape_file, encoding=encoding) as sf:
        shapes, fields, records = sf.shapes(), sf.fields, sf.records()
        heading = [h.lower() for h in list(zip(*fields)).pop(0)[1:]]  # Note 1
        for record, shape in tqdm(zip(records, shapes), desc="Reading Shape File", total=len(records)):
            item = dict(zip(heading, list(record)))
            item['shape'] = shape
            file.append(item)
    return file


def test_directory(path):
    """ Tests a file path and ensures the path exists.  If it does not exist, I will create the path
    :param path: String of a path
    """
    p = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(p):
        os.makedirs(p)
