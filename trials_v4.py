"""Trials with V4 cartogram."""
# simplified: no multiparts, no enclaves!!!

import json
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyproj
import random
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
# from sklearn import manifold

path = "/home/michal/dev/cartograms/cartogram/"


# find neighbours
def _hneighbours(hexagons):
    out = []
    for h in hexagons:
        for i in range(0, 6):
            pointx = round(h[0] + 2 * r * math.sin(math.pi / 3 * i + math.pi / 6), 10)
            pointy = round(h[1] + 2 * r * math.cos(math.pi / 3 * i + math.pi / 6), 10)
            if [pointx, pointy] not in hexagons:
                out.append([pointx, pointy])
    return out


# loss function - number of points outside the polygon
def _hloss(h, polygon):
    out = 7 + 6  # hexagons
    for i in range(0, 6):
        cornerx = h[0] + R * math.sin(math.pi / 3 * i)
        cornery = h[1] + R * math.cos(math.pi / 3 * i)
        if polygon.contains(Point(cornerx, cornery)):
            out = out - 1
        cornerx = h[0] + R / 2 * math.sin(math.pi / 3 * i)
        cornery = h[1] + R / 2 * math.cos(math.pi / 3 * i)
        if polygon.contains(Point(cornerx, cornery)):
            out = out - 1
    if polygon.contains(Point(h[0], h[1])):
        out = out - 1
    return out / (7 + 6)


# loss function - distances of the outside points from polygon
def _hloss2(h, polygon):
    out = 0  # hexagons
    for i in range(0, 6):
        cornerx = h[0] + R * math.sin(math.pi / 3 * i)
        cornery = h[1] + R * math.cos(math.pi / 3 * i)
        out += Point([cornerx, cornery]).distance(polygon)
        cornerx = h[0] + R / 2 * math.sin(math.pi / 3 * i)
        cornery = h[1] + R / 2 * math.cos(math.pi / 3 * i)
        out += Point([cornerx, cornery]).distance(polygon)
    out += Point(h).distance(polygon)
    return out


population = {
    'CZ': 10693939,
    'HU': 9772756,
    'PL': 38383000,
    'SK': 5457926,
    'DE': 83166711,
    'AT': 8902600,
    'UA': 41660982
}
colors = {
    'CZ': 'red',
    'SK': 'blue',
    'HU': 'green',
    'PL': 'pink',
    'DE': 'black',
    'AT': 'orange',
    'UA': 'yellow'
}

with open(path + "maps/v7.topo.json") as fin:
    topo4326 = json.load(fin)

# Transformation from 4326 to 3857 (pseudomercator)
outProj = 'epsg:3857'
inProj = 'epsg:4326'
transformer = pyproj.Transformer.from_crs(inProj, outProj)

# countries and borders
countries3857 = {}
allborders = {}
for geometry in topo4326['objects']['tracts']['geometries']:
    country_id = geometry['properties']['CNTR_ID']
    countries3857[country_id] = {
        'polygon': pd.DataFrame([], columns=['x', 'y'])
    }
    i = 0
    for arc in geometry['arcs'][0][0]:
        line = []
        if arc >= 0:
            arc_id = arc
            iterate = topo4326['arcs'][arc_id]
        else:
            arc_id = -1 * arc - 1
            iterate = reversed(topo4326['arcs'][arc_id])
        for coordinates in iterate:
            item = list(transformer.transform(coordinates[1], coordinates[0]))
            line.append({'x': item[0], 'y': item[1]})
        if arc_id not in allborders.keys():
            allborders[arc_id] = {
                'countries': [],
                'line': pd.DataFrame(line, columns=['x', 'y'])
            }
        allborders[arc_id]['countries'].append(country_id)
        allborders[arc_id]
        if i > 0:
            del line[0]
        countries3857[country_id]['polygon'] = countries3857[country_id]['polygon'].append(line)
        i += 1

# borders length
borders = {}
bordering = []
for i in allborders:
    if len(allborders[i]['countries']) > 1:
        borders[i] = allborders[i]
        borders[i]['length'] = LineString(borders[i]['line'].values.tolist()).length
        bordering.append(allborders[i]['countries'])
        bordering.append([allborders[i]['countries'][1], allborders[i]['countries'][0]])


# area of the original
# https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates/30408825#30408825
def _PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# polygons' areas and centers
for country_id in countries3857:
    countries3857[country_id]['area'] = _PolyArea(countries3857[country_id]['polygon']['x'], countries3857[country_id]['polygon']['y'])
    countries3857[country_id]['center'] = list(list(Polygon(countries3857[country_id]['polygon'].values.tolist()).centroid.coords)[0])

# examples:
# Point(countries3857[country_id]['center']).distance(Polygon(countries3857[country_id]['polygon'].values.tolist()))
# Point(countries3857['CZ']['center']).distance(Polygon(countries3857['SK']['polygon'].values.tolist()))
# Point(countries3857['SK']['center']).distance(Polygon(countries3857['HU']['polygon'].values.tolist()))


# averages, rates
averages = {}
totals = {'area': 0, 'population': 0}
polygons = []
for country_id in countries3857:
    totals['area'] += countries3857[country_id]['area']
    totals['population'] += population[country_id]
    countries3857[country_id]['density'] = population[country_id] / countries3857[country_id]['area']
    polygons.append(Polygon(countries3857[country_id]['polygon'].values.tolist()))
averages['density'] = totals['population'] / totals['area']
for country_id in countries3857:
    countries3857[country_id]['density_rate'] = countries3857[country_id]['density'] / averages['density']

# https://en.wikipedia.org/wiki/Hexagon
R = math.sqrt(2 / 3 / math.sqrt(3))
r = math.cos(math.pi / 6) * R


# population
N = 500000  # 1 hexagon = N people
n = 0
for country_id in countries3857:
    countries3857[country_id]['n'] = round(population[country_id] / N)
    n += countries3857[country_id]['n']

# transformed:
countries = {}
for country_id in countries3857:
    countries[country_id] = {
        'id': country_id,
        'n': countries3857[country_id]['n']
    }

# transformed centers:
totals['polygon'] = unary_union(polygons)
totals['center'] = list(totals['polygon'].centroid.coords[0])
scale = 1 / (math.sqrt(totals['area']) / math.sqrt(n))

# totals['polygon'].area
# r1 = math.sqrt(countries3857['PL']['area'] / math.pi) * scale
# math.pi * r1 * r1
# rscale = math.sqrt(countries['PL']['n'] / math.pi / r1 / r1)
# d1 = Point(countries3857['PL']['center']).distance(Point(totals['center'])) * scale
# d2 = d1 * rscale
# countries['PL']['center'] = [
#     (countries3857['PL']['center'][0] - totals['center'][0]) * scale * rscale,
#     (countries3857['PL']['center'][1] - totals['center'][1]) * scale * rscale
# ]
#
# r1 = math.sqrt(countries3857['CZ']['area'] / math.pi) * scale
# math.pi * r1 * r1
# rscale = math.sqrt(countries['CZ']['n'] / math.pi / r1 / r1)
# d1 = Point(countries3857['CZ']['center']).distance(Point(totals['center'])) * scale
# gap = max(0, d1 - r1)
# d2 = (d1 - r1) * rscale
# bpoint = [
#     (countries3857['CZ']['center'][0] - totals['center'][0]) * scale * (gap / d1),
#     (countries3857['CZ']['center'][1] - totals['center'][1]) * scale * (gap / d1)
# ]
# countries['CZ']['center'] = [
#     ((countries3857['CZ']['center'][0] - totals['center'][0]) * scale - bpoint[0]) * rscale + bpoint[0],
#     ((countries3857['CZ']['center'][1] - totals['center'][1]) * scale - bpoint[1]) * rscale + bpoint[1],
# ]

for country_id in countries3857:
    r1 = math.sqrt(countries3857[country_id]['area'] / math.pi) * scale
    rscale = math.sqrt(countries[country_id]['n'] / math.pi / r1 / r1)
    d1 = Point(countries3857[country_id]['center']).distance(Point(totals['center'])) * scale
    gap = max(0, d1 - r1)
    bpoint = [
        (countries3857[country_id]['center'][0] - totals['center'][0]) * scale * (gap / d1),
        (countries3857[country_id]['center'][1] - totals['center'][1]) * scale * (gap / d1)
    ]
    rawcenter = [
        (countries3857[country_id]['center'][0] - totals['center'][0]) * scale,
        (countries3857[country_id]['center'][1] - totals['center'][1]) * scale
    ]
    countries[country_id]['center'] = [
        ((countries3857[country_id]['center'][0] - totals['center'][0]) * scale - bpoint[0]) * rscale + bpoint[0],
        ((countries3857[country_id]['center'][1] - totals['center'][1]) * scale - bpoint[1]) * rscale + bpoint[1],
    ]
    countries[country_id]['shift'] = [
        countries[country_id]['center'][0] - rawcenter[0],
        countries[country_id]['center'][1] - rawcenter[1]
    ]

for country_id in countries3857:
    countries[country_id]['polygon'] = pd.DataFrame([], columns=['x', 'y'])
    countries[country_id]['polygon']['x'] = (countries3857[country_id]['polygon']['x'] - totals['center'][0]) * scale + countries[country_id]['shift'][0]
    countries[country_id]['polygon']['y'] = (countries3857[country_id]['polygon']['y'] - totals['center'][1]) * scale + countries[country_id]['shift'][1]

fig = go.Figure()
for country_id in countries:
    xs = list(countries[country_id]['polygon']['x'])
    ys = list(countries[country_id]['polygon']['y'])
    fig.add_trace(go.Scatter(x=xs, y=ys))
fig.show()


# find closest center
def _get_closest_hexagon(p):
    p = Point(p)
    not_found = True
    last = [[0, 0]]
    tried = []
    while not_found:
        for h in last:
            corners = []
            for i in range(0, 7):
                cornerx = h[0] + R * math.sin(math.pi / 3 * i)
                cornery = h[1] + R * math.cos(math.pi / 3 * i)
                corners.append([cornerx, cornery])
            if Polygon(corners).contains(p):
                not_found = False
                closest = h
            tried = tried + [h]
        last = _hneighbours(tried)
    return closest


# covering by hexagons
ids_list = []
for country_id in countries:
    ids_list = ids_list + [country_id] * (countries[country_id]['n'] - 1)
random.shuffle(ids_list)

hexagons = {}
h_all = []
hx = []
hy = []
for country_id in countries:
    nh = _get_closest_hexagon(countries[country_id]['center'])
    hexagons[country_id] = [nh]
    h_all.append(nh)
    hx.append(nh[0])
    hy.append(nh[1])
    countries[country_id]['Polygon'] = Polygon(countries[country_id]['polygon'].values.tolist())

i = 0
for country_id in ids_list:
    neighbours = _hneighbours(hexagons[country_id])
    best_neighbour = None
    for neighbour in neighbours:
        if neighbour not in h_all:
            best_neighbour = neighbour
            break
    if not best_neighbour:
        print('Oh well, no more place for: ' + country_id)
    else:
        best_loss = _hloss(best_neighbour, countries[country_id]['Polygon'])
        for neighbour in neighbours:
            if neighbour not in h_all:
                neighbour_loss = _hloss(neighbour, countries[country_id]['Polygon'])
                if neighbour_loss < best_loss:
                    best_neighbour = neighbour
                    best_loss = neighbour_loss
        hexagons[country_id].append(best_neighbour)
        h_all.append(best_neighbour)
        hx.append(best_neighbour[0])
        hy.append(best_neighbour[1])
        print(i, best_neighbour)
        i += 1


fig = go.Figure()
# for country_id in countries:
#     xs = list(countries[country_id]['polygon']['x'])
#     ys = list(countries[country_id]['polygon']['y'])
#     fig.add_trace(go.Scatter(x=xs, y=ys, marker_color=colors[country_id]))
for country_id in countries:
    for h in hexagons[country_id]:
        pth = ""
        for i in range(0, 7):
            cornerx = h[0] + R * math.sin(math.pi / 3 * i)
            cornery = h[1] + R * math.cos(math.pi / 3 * i)
            if i == 0:
                pth += "M "
            else:
                pth += "L "
            pth += str(cornerx) + " " + str(cornery) + " "
        pth += "Z"
        fig.add_shape(
            type="path",
            path=pth,
            fillcolor=colors[country_id],
            opacity=0.5,
            line=dict(
                color=colors[country_id],
                width=2,
            )
        )
fig.add_trace(go.Scatter(
    x=hx,
    y=hy,
    text=list(range(0, n + 1)),
    mode="text",
    textposition="middle center"
))
fig.update_layout(
    autosize=False,
    width=1200,
    height=700
)
fig.show()
fig.write_image(path + "images/v7_hexagons_500000_example_without.png")



# # MDS for centers:
# distances = []
# # pd.DataFrame([], columns=['x', 'y']
# for c_id1 in countries:
#     row = []
#     for c_id2 in countries:
#         if c_id1 == c_id2:
#             row.append(0)
#         else:
#             if [c_id1, c_id2] in bordering:
#                 dist = math.sqrt(countries[c_id1]['n']) + math.sqrt(countries[c_id2]['n'])
#             else:
#                 d = Point(countries3857[c_id1]['center']).distance(Point(countries3857[c_id2]['center']))
#                 r1 = math.sqrt(countries3857[c_id1]['area']) / math.pi
#                 r2 = math.sqrt(countries3857[c_id2]['area']) / math.pi
#                 gap = max((d - r1 - r2) * (d - r1 - r2) * averages['density'] / N, 0)
#                 dist = math.sqrt(countries[c_id1]['n']) + math.sqrt(countries[c_id2]['n']) + gap
#             row.append(dist)
#     distances.append(row)
#
#
# # MDS
# countries.keys()
# mds = manifold.MDS(dissimilarity='precomputed')
# results = mds.fit(distances)
# coords = results.embedding_

# d = Point(countries3857['PL']['center']).distance(Point(countries3857['CZ']['center']))
# r1 = math.sqrt(countries3857['PL']['area']) / math.pi
# r2 = math.sqrt(countries3857['CZ']['area']) / math.pi
# gap = (d - r1 - r2) * (d - r1 - r2) * averages['density'] / N
# dist = math.sqrt(countries3857['PL']['n']) + math.sqrt(countries3857['CZ']['n']) + gap


# note: simplified for no enclaves only (no "Vatican within Italy")
# def _polygon_borders_from_topo(geometry, topo):
#     out = []
#     for polygon in geometry['arcs']:
#         for part in polygon[0]:
#             nothing = 0
