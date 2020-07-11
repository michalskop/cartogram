"""Converts json file into ndjson."""

import json
import ndjson

path = "/home/michal/dev/cartograms/algorithmic/maps/"
with open(path + "cz.geo.json") as fin:
    data = json.load(fin)

with open(path + "cz.geo.ndjson", "w") as fout:
    ndjson.dump(data, fout)
