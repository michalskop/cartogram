"""Trials with V7 cartogram."""
# simplified: no multiparts, no enclaves!!!

import json
import math
import plotly.graph_objects as go
import pyproj
import random
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
# from sklearn import manifold

path = "/home/michal/dev/cartograms/cartogram/"

# https://en.wikipedia.org/wiki/Hexagon
R = math.sqrt(2 / 3 / math.sqrt(3))
r = math.cos(math.pi / 6) * R


# find neighbours
# @hexagons list of Points
def _hneighbours(hexagons):
    out = []
    for h in hexagons:
        for i in range(0, 6):
            pointx = h.x + 2 * r * math.sin(math.pi / 3 * i + math.pi / 6)
            pointy = h.y + 2 * r * math.cos(math.pi / 3 * i + math.pi / 6)
            neighbour = Point([pointx, pointy])
            if neighbour not in hexagons:
                out.append(neighbour)
    return out


# loss function - distances of the outside points from polygon
# @h center of hexagon Point
# @polygon the Polygon
def _hloss(h, polygon):
    out = 0  # hexagons
    for i in range(0, 6):
        corner_x = h.x + R * math.sin(math.pi / 3 * i)
        corner_y = h.y + R * math.cos(math.pi / 3 * i)
        out += Point([corner_x, corner_y]).distance(polygon)
        half_corner_x = h.x + R / 2 * math.sin(math.pi / 3 * i)
        half_corner_y = h.y + R / 2 * math.cos(math.pi / 3 * i)
        out += Point([half_corner_x, half_corner_y]).distance(polygon)
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
population = {
    'CZ020': 2678000,
    'CZ031': 642000,
    'CZ032': 585000,
    'CZ041': 295000,
    'CZ042': 821000,
    'CZ051': 442000,
    'CZ052': 551000,
    'CZ053': 520000,
    'CZ063': 509000,
    'CZ064': 1189000,
    'CZ071': 633000,
    'CZ072': 583000,
    'CZ080': 1203000
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
colors = {
    'CZ020': '#e9ecef',
    'CZ031': '#138496',
    'CZ032': '#E95420',
    'CZ041': '#ecaa1b',
    'CZ042': '#abdde5',
    'CZ051': '#f3b2ae',
    'CZ052': '#2f973e',
    'CZ053': '#f7bdaa',
    'CZ063': '#c7291e',
    'CZ064': '#772953',
    'CZ071': '#cfb3c3',
    'CZ072': '#dcd9d6',
    'CZ080': '#9c948a'
}

with open(path + "maps/v7.topo.json") as fin:
    topo4326 = json.load(fin)
with open(path + "maps/kraje-simple-topo_noprague.json") as fin:
    topo4326 = json.load(fin)

# Transformation from 4326 to 3857 (pseudomercator)
outProj = 'epsg:3857'
inProj = 'epsg:4326'
transformer = pyproj.Transformer.from_crs(inProj, outProj)

# countries and borders
countries3857 = {}
allborders = {}
for geometry in topo4326['objects']['tracts']['geometries']:
    # country_id = geometry['properties']['CNTR_ID']
    country_id = geometry['properties']['NUTS3_KOD']
    polygons = []
    for part in geometry['arcs']:
        i = 0
        polygon_line = []
        # for arc in part[0]: # world
        for arc in part:    # kraje
            line = []
            if arc >= 0:
                arc_id = arc
                iterate = topo4326['arcs'][arc_id]
            else:
                arc_id = -1 * arc - 1
                iterate = reversed(topo4326['arcs'][arc_id])
            for coordinates in iterate:
                item = list(transformer.transform(coordinates[1], coordinates[0]))
                line.append(item)
            if arc_id not in allborders.keys():
                allborders[arc_id] = {
                    'countries': [],
                    'line': LineString(line)
                }
            allborders[arc_id]['countries'].append(country_id)
            # allborders[arc_id]
            if i > 0:
                del line[0]
            polygon_line = polygon_line + line
            i += 1
        polygons.append(Polygon(polygon_line))
    countries3857[country_id] = {
        'multipolygon': MultiPolygon(polygons)
    }


# borders length
borders = {}
bordering = []
for i in allborders:
    if len(allborders[i]['countries']) > 1:
        borders[i] = allborders[i]
        borders[i]['length'] = borders[i]['line'].length
        bordering.append(allborders[i]['countries'])
        bordering.append([allborders[i]['countries'][1], allborders[i]['countries'][0]])

# multipolygons' areas and centers
for country_id in countries3857:
    countries3857[country_id]['area'] = countries3857[country_id]['multipolygon'].area
    countries3857[country_id]['center'] = countries3857[country_id]['multipolygon'].centroid

# totals/averages, density
totals = {'area': 0, 'population': 0, 'density': 0}
for country_id in countries3857:
    totals['area'] += countries3857[country_id]['area']
    totals['population'] += population[country_id]
    countries3857[country_id]['density'] = population[country_id] / countries3857[country_id]['area']
totals['density'] = totals['population'] / totals['area']

# population
N = 50000  # 1 hexagon = N people
n = 0
for country_id in countries3857:
    countries3857[country_id]['n'] = round(population[country_id] / N)
    n += countries3857[country_id]['n']

# transformed:
countries = {}
for country_id in countries3857:
    countries[country_id] = {
        'id': country_id,
        'n': countries3857[country_id]['n'],
        'scale': math.sqrt(countries3857[country_id]['density'] / totals['density'])
    }

# centers:
multipolygons = []
for country_id in countries3857:
    multipolygons.append(countries3857[country_id]['multipolygon'])
totals['multipolygons'] = unary_union(multipolygons)
totals['raw_center'] = totals['multipolygons'].centroid
x = 0
y = 0
for country_id in countries3857:
    x += countries3857[country_id]['center'].x * population[country_id]
    y += countries3857[country_id]['center'].y * population[country_id]
totals['center'] = Point([x / totals['population'], y / totals['population']])

# scale
scale = math.sqrt(n) / math.sqrt(totals['area'])

# transformed centers
for country_id in countries:
    # scale and move around [0,0]:
    raw_center = affinity.translate(affinity.scale(countries3857[country_id]['center'], xfact=scale, yfact=scale, origin=totals['center']), xoff=-1 * totals['center'].x, yoff=-1 * totals['center'].y)
    raw_multipolygon = affinity.translate(affinity.scale(countries3857[country_id]['multipolygon'], xfact=scale, yfact=scale, origin=totals['center']), xoff=-1 * totals['center'].x, yoff=-1 * totals['center'].y)

    # find boundary point:
    raw_area = countries3857[country_id]['area'] * scale * scale
    r1 = math.sqrt(raw_area / math.pi)
    d = Point(raw_center).distance(Point([0, 0]))
    if d > 0:
        bpointx = raw_center.x * (1 - r1 / d)
        bpointy = raw_center.y * (1 - r1 / d)
        bpoint = Point([bpointx, bpointy])
    else:
        bpoint = Point([0, 0])

    # country scale around border point
    countries[country_id]['center'] = affinity.scale(raw_center, xfact=countries[country_id]['scale'], yfact=countries[country_id]['scale'], origin=bpoint)
    countries[country_id]['multipolygon'] = affinity.scale(raw_multipolygon, xfact=countries[country_id]['scale'], yfact=countries[country_id]['scale'], origin=bpoint)

fig = go.Figure()
for country_id in countries:
    for polygon in countries[country_id]['multipolygon']:
        x, y = polygon.exterior.coords.xy
        fig.add_trace(go.Scatter(x=list(x), y=list(y)))
fig.show()


# find closest center
def _get_closest_hexagon(p):
    not_found = True
    x = round(p.x / 2 / r) * 2 * r
    y = round(p.y / 3 / R) * 3 * R
    last = [Point([x, y])]
    tried = []
    while not_found:
        for h in last:
            corners = []
            for i in range(0, 7):
                cornerx = h.x + R * math.sin(math.pi / 3 * i)
                cornery = h.y + R * math.cos(math.pi / 3 * i)
                corners.append(Point([cornerx, cornery]))
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
h_all_rounded = []
hx = []
hy = []
for country_id in countries:
    nh = _get_closest_hexagon(countries[country_id]['center'])
    hexagons[country_id] = [nh]
    rounded_nh = [round(nh.x, 2), round(nh.y, 2)]
    h_all_rounded.append(rounded_nh)
    hx.append(nh.x)
    hy.append(nh.y)

i = 0
for country_id in ids_list:
    neighbours = _hneighbours(hexagons[country_id])
    random.shuffle(neighbours)
    best_neighbour = None
    for neighbour in neighbours:
        rounded_neighbour = [round(neighbour.x, 2), round(neighbour.y, 2)]
        if rounded_neighbour not in h_all_rounded:
            best_neighbour = neighbour
            break
    if not best_neighbour:
        print('Oh well, no more place for: ' + country_id)
    else:
        best_loss = _hloss(best_neighbour, countries[country_id]['multipolygon'])
        for neighbour in neighbours:
            rounded_neighbour = [round(neighbour.x, 2), round(neighbour.y, 2)]
            if rounded_neighbour not in h_all_rounded:
                neighbour_loss = _hloss(neighbour, countries[country_id]['multipolygon'])
                if neighbour_loss == 0:
                    best_neighbour = neighbour
                    best_loss = neighbour_loss
                    break
                if neighbour_loss < best_loss:
                    best_neighbour = neighbour
                    best_loss = neighbour_loss
        hexagons[country_id].append(best_neighbour)
        rounded_best_neighbour = [round(best_neighbour.x, 2), round(best_neighbour.y, 2)]
        h_all_rounded.append(rounded_best_neighbour)
        hx.append(best_neighbour.x)
        hy.append(best_neighbour.y)
        print(i, best_neighbour.x)
        i += 1

fig = go.Figure()
# for country_id in countries:
#     for polygon in countries[country_id]['multipolygon']:
#         x, y = polygon.exterior.coords.xy
#         fig.add_trace(go.Scatter(x=list(x), y=list(y), marker_color=colors[country_id]))
for country_id in countries:
    for h in hexagons[country_id]:
        pth = ""
        for i in range(0, 7):
            cornerx = h.x + R * math.sin(math.pi / 3 * i)
            cornery = h.y + R * math.cos(math.pi / 3 * i)
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
    # text=list(range(0, n + 1)),
    mode="text",
    textposition="middle center"
))
fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    autosize=False,
    width=1200,
    height=700,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False),
)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()
fig.write_image(path + "images/cz_kraje_50000.png")

#
#
# a = Point(0, 0)
# b = Point(1, 1)
# (b - a).y
#
#
# list(countries['DE']['raw_multipolygon'][0].exterior.coords)
# x, y = countries['DE']['multipolygon'][0].exterior.coords.xy
# list(countries['DE']['center'].coords)
