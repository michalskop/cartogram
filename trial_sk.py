"""Trials with SK cartogram."""
# simplified:no enclaves!!!

import copy
import hexutil
import json
import math
import plotly.graph_objects as go
import pyproj
import random
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

path = "/home/michal/dev/cartograms/cartogram/"

# set up hexagons grid
# https://en.wikipedia.org/wiki/Hexagon
R = math.sqrt(2 / 3 / math.sqrt(3))
r = math.cos(math.pi / 6) * R
# https://github.com/stephanh42/hexutil
hexgrid = hexutil.HexGrid(width=r, height=R / 2)


# loss function - distances of the corners and center from polygon and continent
# @h center of hexagon Point
# @polygon the Polygon
def _loss(h, polygon):
    loss = 0
    corners = hexgrid.corners(h)
    for corner in corners:
        p = Point(corner)
        loss += p.distance(polygon)
        loss += p.distance(continent['polygon'])
    loss += Point(hexgrid.center(h)).distance(polygon)
    return loss


# loss function - distance for inner point
# @h center of hexagon Point
# @polygons - object of the Polygons
def _loss_overlap(h, polygons, except_id):
    loss = 0
    for p_id in polygons:
        if p_id != except_id:
            polygon = polygons[p_id]['polygon']
            c = Point(hexgrid.center(h))
            if c.distance(polygon) == 0:
                loss += c.distance(polygon.exterior)
    return loss


# create a svg path for the hexagon
def _create_hexagon_path(h):
    path = ""
    corners = hexgrid.corners(h)
    corners.append(corners[0])
    i = 0
    for corner in corners:
        if i == 0:
            path += "M "
        else:
            path += "L "
        path += str(corner[0]) + " " + str(corner[1]) + " "
        i += 1
    path += "Z"
    return path


# find neighbours
# @hexagons list of Hexs
def _hneighbours(hexagons):
    out = []
    for h in hexagons:
        neighbours = h.neighbours()
        for neighbour in neighbours:
            if neighbour not in hexagons:
                out.append(neighbour)
    return out


# destroy lakes
# is landlocked
def _is_landlocked(h, hexagons):
    neighbours = h.neighbours()
    rocks = 0
    for neighbour in neighbours:
        for country_id in hexagons:
            if neighbour in hexagons[country_id]:
                rocks += 1
                break
    if rocks == 6:
        return True
    else:
        return False


# is lake (size 1, with 6 beaches)
def _is_lake(h, hexagons):
    is_water = True
    for country_id in hexagons:
        if h in hexagons[country_id]:
            is_water = False
    if not is_water:
        return False
    return _is_landlocked(h, hexagons)


population = {
    'SK010': 669592,
    'SK021': 564917,
    'SK022': 584917,
    'SK023': 674306,
    'SK031': 691509,
    'SK032': 645276,
    'SK041': 826244,
    'SK042': 801460
}
colors = {
    'SK010': '#138496',
    'SK021': '#E95420',
    'SK022': '#ecaa1b',
    'SK023': '#c7291e',
    'SK031': '#772953',
    'SK032': '#2f973e',
    'SK041': '#9c948a',
    'SK042': '#b4e3bb'
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
    'CZ020': '#b4e3bb',
    'CZ031': '#138496',
    'CZ032': '#E95420',
    'CZ041': '#ecaa1b',
    'CZ042': '#abdde5',
    'CZ051': '#f3b2ae',
    'CZ052': '#dcd9d6',
    'CZ053': '#f7bdaa',
    'CZ063': '#c7291e',
    'CZ064': '#772953',
    'CZ071': '#cfb3c3',
    'CZ072': '#2f973e',
    'CZ080': '#9c948a'
}

# Transformation from 4326 to 3857 (pseudomercator)
outProj = 'epsg:3857'
inProj = 'epsg:4326'
transformer = pyproj.Transformer.from_crs(inProj, outProj)

with open(path + "maps/sk_kraje.geojson") as fin:
    geo4326 = json.load(fin)
with open(path + "maps/cz_kraje_noprague.geojson") as fin:
    geo4326 = json.load(fin)

countries3857 = {}
for feature in geo4326['features']:
    country_id = feature['properties']['NUTS_ID']
    countries3857[country_id] = {}
    line = []
    for coordinates in feature['geometry']['coordinates'][0][0]:
        item = list(transformer.transform(coordinates[1], coordinates[0]))
        line.append(item)
    countries3857[country_id]['polygon'] = Polygon(line)

# polygons' areas and centers
for country_id in countries3857:
    countries3857[country_id]['area'] = countries3857[country_id]['polygon'].area
    countries3857[country_id]['center'] = countries3857[country_id]['polygon'].centroid

# totals/averages, density
totals = {'area': 0, 'population': 0, 'density': 0}
for country_id in countries3857:
    totals['area'] += countries3857[country_id]['area']
    totals['population'] += population[country_id]
    countries3857[country_id]['density'] = population[country_id] / countries3857[country_id]['area']
totals['density'] = totals['population'] / totals['area']

# population
N = 20000  # 1 hexagon = N people
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
polygons = []
for country_id in countries3857:
    polygons.append(countries3857[country_id]['polygon'])
totals['polygon'] = unary_union(polygons)
totals['raw_center'] = totals['polygon'].centroid
x = 0
y = 0
for country_id in countries3857:
    x += countries3857[country_id]['center'].x * population[country_id]
    y += countries3857[country_id]['center'].y * population[country_id]
totals['center'] = Point([x / totals['population'], y / totals['population']])

# scale
scale = math.sqrt(n) / math.sqrt(totals['area'])

# continent
continent = {
    'center': Point(0, 0),
    'scale': 1,
    'n': n,
    'area': n,
    'polygon': affinity.translate(affinity.scale(totals['polygon'], xfact=scale, yfact=scale, origin=totals['center']), xoff=-1 * totals['center'].x, yoff=-1 * totals['center'].y)
}

# transformed centers and polygons
for country_id in countries:
    # scale and move around [0,0]:
    raw_center = affinity.translate(affinity.scale(countries3857[country_id]['center'], xfact=scale, yfact=scale, origin=totals['center']), xoff=-1 * totals['center'].x, yoff=-1 * totals['center'].y)
    raw_polygon = affinity.translate(affinity.scale(countries3857[country_id]['polygon'], xfact=scale, yfact=scale, origin=totals['center']), xoff=-1 * totals['center'].x, yoff=-1 * totals['center'].y)

    countries[country_id]['center'] = affinity.scale(raw_center, xfact=countries[country_id]['scale'], yfact=countries[country_id]['scale'], origin=raw_center)
    countries[country_id]['polygon'] = affinity.scale(raw_polygon, xfact=countries[country_id]['scale'], yfact=countries[country_id]['scale'], origin=raw_center)

    countries[country_id]['area'] = countries[country_id]['polygon'].area

fig = go.Figure()
x, y = continent['polygon'].exterior.coords.xy
fig.add_trace(go.Scatter(x=list(x), y=list(y)))
for country_id in countries:
    x, y = countries[country_id]['polygon'].exterior.coords.xy
    fig.add_trace(go.Scatter(x=list(x), y=list(y)))
fig.show()

for country_id in countries:
    countries[country_id]['gravity_center'] = countries[country_id]['center']
    countries[country_id]['weight'] = population[country_id] / totals['population']

cw = 2
cxw = 1
epsilon = 0.000001
maxiter = 1000
last_moves = 1000
for i in range(0, maxiter):
    # tick
    moves = 0
    for country_id in countries:
        c = countries[country_id]
        mx = 0
        my = 0
        # continent
        outside = (1 - c['polygon'].intersection(continent['polygon']).area / c['polygon'].area)
        d = c['center'].distance(Point(0, 0))
        mx += -1 * outside * outside * c['weight'] * cw * c['center'].x / d
        my += -1 * outside * outside * c['weight'] * cw * c['center'].y / d
        # countries
        for cid in countries:
            if cid != country_id:
                # cid = 'SK041'
                cx = countries[cid]
                overlap = c['polygon'].intersection(cx['polygon']).area / (c['area'] + cx['area'])
                d = c['center'].distance(cx['center'])
                mx += (c['center'].x - cx['center'].x) * overlap * overlap * (c['weight'] / (c['weight'] + cx['weight'])) * cxw
                my += (c['center'].y - cx['center'].y) * overlap * overlap * (c['weight'] / (c['weight'] + cx['weight'])) * cxw
        c['new_center'] = Point(c['center'].x + mx, c['center'].y + my)
        c['move'] = [mx, my]
        moves += math.sqrt(mx * mx + my * my)
    for country_id in countries:
        c = countries[country_id]
        c['center'] = c['new_center']
        c['polygon'] = affinity.translate(c['polygon'], xoff=c['move'][0], yoff=c['move'][1])

    if (last_moves - moves) < epsilon:
        if (i / 10) == round(i / 10):
            print(i, last_moves)
        break
    last_moves = moves
print(i)


fig = go.Figure()
x, y = continent['polygon'].exterior.coords.xy
fig.add_trace(go.Scatter(x=list(x), y=list(y)))
for country_id in countries:
    x, y = countries[country_id]['polygon'].exterior.coords.xy
    fig.add_trace(go.Scatter(x=list(x), y=list(y)))
fig.show()







# START SIMULATION
# covering by hexagons
# random order of covering
ids_list = []
for country_id in countries:
    ids_list = ids_list + [country_id] * (countries[country_id]['n'] - 1)
random.shuffle(ids_list)

# central hexagon for each country
hexagons = {}
h_all = []
hx = []
hy = []
hxy = []
for country_id in countries:
    country = countries[country_id]
    h = hexgrid.hex_at_coordinate(country['center'].x, country['center'].y)
    hexagons[country_id] = [h]
    h_all.append(h)
    c = hexgrid.center(h)
    hx.append(c[0])
    hy.append(c[1])
    hxy.append(str(round(h.x)) + ',' + str(round(h.y)))

# covering by hexagons
i = 0
for country_id in ids_list:
    neighbours = _hneighbours(hexagons[country_id])
    random.shuffle(neighbours)
    best_neighbour = None
    for neighbour in neighbours:
        if neighbour not in h_all:
            best_neighbour = neighbour
            break
    if not best_neighbour:
        print('Oh mine, no more place for: ' + country_id)
        best_neighbour = hexutil.Hex(1000, 1000)
    else:
        best_loss = _loss(best_neighbour, countries[country_id]['polygon'])
        for neighbour in neighbours:
            if neighbour not in h_all:
                loss = _loss(neighbour, countries[country_id]['polygon'])
                loss_overlap = _loss_overlap(neighbour, countries, country_id)
                neighbour_loss = loss + 7 * loss_overlap
                if neighbour_loss == 0:
                    best_neighbour = neighbour
                    best_loss = neighbour_loss
                    break
                if neighbour_loss < best_loss:
                    best_neighbour = neighbour
                    best_loss = neighbour_loss
        hexagons[country_id].append(best_neighbour)
        h_all.append(best_neighbour)
        c = hexgrid.center(best_neighbour)
        hx.append(c[0])
        hy.append(c[1])
        hxy.append(str(round(best_neighbour.x)) + ',' + str(round(best_neighbour.y)))
        # print(i, best_neighbour.x, best_neighbour.y)
        i += 1
print(i)


# _loss_overlap(hexutil.Hex(0, 0), countries, country_id)

fig = go.Figure()
x, y = continent['polygon'].exterior.coords.xy
fig.add_trace(go.Scatter(x=list(x), y=list(y)))
# for country_id in countries:
#     for polygon in countries[country_id]['polygon']:
#         x, y = polygon.exterior.coords.xy
#         fig.add_trace(go.Scatter(x=list(x), y=list(y), marker_color=colors[country_id]))
for country_id in countries:
    for h in hexagons[country_id]:
        pth = _create_hexagon_path(h)
        fig.add_shape(
            type="path",
            path=pth,
            fillcolor=colors[country_id],
            opacity=0.75,
            line=dict(
                color=colors[country_id],
                width=2,
            )
        )
# for country_id in countries:
#     x = []
#     y = []
#     for h in hexagons[country_id]:
#         c = hexgrid.center(h)
#         x.append(c[0])
#         y.append(c[1])
#         fig.add_trace(go.Scatter(x=list(x), y=list(y), marker_color=colors[country_id]))
fig.add_trace(go.Scatter(
    x=hx,
    y=hy,
    text=hxy,
    mode="text",
    textposition="middle center"
))
fig.update_layout(
    # paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    autosize=False,
    width=900,
    height=550,
    # xaxis_showgrid=False,
    # yaxis_showgrid=False,
    # xaxis=dict(showgrid=False, zeroline=False),
    # yaxis=dict(showgrid=False, zeroline=False),
)
# fig.update_xaxes(showticklabels=False)
# fig.update_yaxes(showticklabels=False)
fig.show()


# shaker+
def _shaker():
    switched = 0
    for country_id in hexagons:
        for i in reversed(range(-1 * len(hexagons[country_id]), 0)):
            h = hexagons[country_id][i]
            best_improvement = 0
            switch = None
            loss = _loss(h, countries[country_id]['polygon'])
            if loss == 0:
                break
            neighbours = _hneighbours(hexagons[country_id])
            for neighbour in neighbours:
                country_id2 = None
                for cid2 in hexagons:
                    if country_id != cid2:
                        if neighbour in hexagons[cid2]:
                            country_id2 = cid2
                            j = hexagons[cid2].index(neighbour)
                            break
                if country_id2:
                    h2 = neighbour
                    loss2 = _loss(h2, countries[country_id2]['polygon'])
                    rloss = _loss(h, countries[country_id2]['polygon'])
                    rloss2 = _loss(h2, countries[country_id]['polygon'])
                    improvement = (loss + loss2) - (rloss + rloss2)
                    if improvement > 0 and improvement > best_improvement:
                        nbrs = h.neighbours()
                        exclave = True
                        for nbr in nbrs:
                            if nbr in hexagons[country_id2]:
                                exclave = False
                        if not exclave:
                            best_improvement = improvement
                            switch = [h, h2, country_id2]
                else:
                    h2 = neighbour
                    rloss2 = _loss(h2, countries[country_id]['polygon'])
                    improvement = loss - rloss2
                    if improvement > 0 and improvement > best_improvement and not _is_landlocked(h, hexagons):
                        best_improvement = improvement
                        switch = [h, h2, country_id2]
            if switch:
                print(country_id, i, switch)
                switched += 1
                if switch[2]:
                    j = hexagons[switch[2]].index(switch[1])
                    hexagons[switch[2]][j] = switch[0]
                hexagons[country_id][i] = switch[1]
    return switched


def _lakes():
    lakes = []
    for country_id in hexagons:
        neighbours = _hneighbours(hexagons[country_id])
        for neighbour in neighbours:
            if neighbour not in lakes and _is_lake(neighbour, hexagons):
                lakes.append(neighbour)

    for lake in lakes:
        hxcp = copy.deepcopy(hexagons)
        hxcp['__lakes__'] = lakes
        cids = []
        for country_id in hexagons:
            if lake in _hneighbours(hexagons[country_id]):
                cids.append(country_id)
        max_loss = 0
        switch = None
        for country_id in cids:
            for h in hexagons[country_id]:
                loss = _loss(h, countries[country_id]['polygon'])
                if loss > max_loss and (not _is_landlocked(h, hxcp)):
                    max_loss = loss
                    switch = [h, country_id]
        if switch:
            print(switch)
            j = hexagons[switch[1]].index(switch[0])
            hexagons[switch[1]][j] = lake
        del hxcp


i = 0
switching = True
while switching:
    print("switching " + str(i))
    s = _shaker()
    _lakes()
    i += 1
    if s == 0 or i > 5:
        switching = False

final_loss = 0



# hexagons_object = {}
# for country_id in countries:
#     country = countries[country_id]
#     hexagons_object[country_id] = []
#     for h in hexagons[country_id]:
#         hexagons_object[country_id].append([round(h.x), round(h.y)])
#
# with open(path + "hexagons/cz_kraje_50000.json", "w") as fout:
#     json.dump(hexagons_object, fout)
#
# with open(path + "hexagons/cz_kraje_50000.json") as fin:
#     hexagons_loaded = json.load(fin)
#
# hexagons_manually = {}
# for country_id in hexagons_loaded:
#     hexagons_manually[country_id] = []
#     for row in hexagons_loaded[country_id]:
#         hexagons_manually[country_id].append(hexutil.Hex(row[0], row[1]))

fig = go.Figure()
# x, y = continent['polygon'].exterior.coords.xy
# fig.add_trace(go.Scatter(x=list(x), y=list(y)))
# for country_id in countries:
#     for polygon in countries[country_id]['polygon']:
#         x, y = polygon.exterior.coords.xy
#         fig.add_trace(go.Scatter(x=list(x), y=list(y), marker_color=colors[country_id]))
for country_id in countries:
    for h in hexagons[country_id]:
        pth = _create_hexagon_path(h)
        fig.add_shape(
            type="path",
            path=pth,
            fillcolor=colors[country_id],
            opacity=0.75,
            line=dict(
                color=colors[country_id],
                width=2,
            )
        )
# for country_id in countries:
#     x = []
#     y = []
#     for h in hexagons[country_id]:
#         c = hexgrid.center(h)
#         x.append(c[0])
#         y.append(c[1])
#         fig.add_trace(go.Scatter(x=list(x), y=list(y), marker_color=colors[country_id]))
fig.add_trace(go.Scatter(
    x=hx,
    y=hy,
    # text=hxy,
    mode="text",
    textposition="middle center"
))
fig.update_layout(
    # paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    autosize=False,
    width=1000,
    height=550,
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False),
)
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.show()
