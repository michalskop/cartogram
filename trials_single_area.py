"""Trials with cartogram."""
# https://github.com/pyproj4/pyproj

import json
import math
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pyproj
from shapely.geometry import Polygon, Point

path = "/home/michal/dev/cartograms/cartogram/"


# Transformation from 4326 to 3857 (pseudomercator)
outProj = 'epsg:3857'
inProj = 'epsg:4326'
transformer = pyproj.Transformer.from_crs(inProj, outProj)

with open(path + "maps/cz-simple.json") as fin:
    geo4326 = json.load(fin)

geo3857 = []
x = []
y = []
for feature in geo4326['features']:
    for coordinates in feature['geometry']['coordinates'][0][0]:
        item = list(transformer.transform(coordinates[1], coordinates[0]))
        geo3857.append(item)
        x.append(item[0])
        y.append(item[1])

fig = px.scatter(x=x, y=y)
fig.show()


# area of the original
# https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates/30408825#30408825
def _PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


area = _PolyArea(x, y)

# centre
polygon3857 = Polygon(geo3857)
centre3857 = list(polygon3857.centroid.coords)[0]

fig = px.scatter(x=x + [centre3857[0]], y=y + [centre3857[1]])
fig.show()


# centre and rescale
<<<<<<< HEAD
n = 20
=======
n = 200
>>>>>>> 8dccf2b8160df843fae09f5b7bf0325e12d22066
scale = 1 / (math.sqrt(area) / math.sqrt(n))
xs = []
ys = []
xys = []
i = 0
for c in x:
    nx = (c - centre3857[0]) * scale
    xs.append(nx)
    ny = (y[i] - centre3857[1]) * scale
    ys.append(ny)
    xys.append([nx, ny])
    i += 1
_PolyArea(xs, ys)
print(xys[0], xys[-1])

fig = px.scatter(x=xs, y=ys)
fig.show()

polygon = Polygon(xys)
polygon.contains(Point([0, 0]))


# covering the map by squares / hexagons - step by step
# loss function squares
def _loss(point, polygon):
    out = 9  # squares
    for i in range(-1, 2):
        for j in range(-1, 2):
            if polygon.contains(Point(point[0] + i / 2, point[1] + j / 2)):
                out = out - 1
    return out / 9


# find neighbours
def _neighbours(squares):
    out = []
    for s in squares:
        for i in range(-1, 2):
            for j in range(-1, 2):
                if not ((i == 0) and (j == 0)):
                    if [s[0] + i, s[1] + j] not in squares:
                        out.append([s[0] + i, s[1] + j])
    return out


# examples
_loss([12, 0], polygon)

_neighbours([[0, 0], [0, 1]])

# https://en.wikipedia.org/wiki/Hexagon
R = math.sqrt(2 / 3 / math.sqrt(3))
r = math.cos(math.pi / 6) * R


# loss function hexagons
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


# find neighbours hexagons
def _hneighbours(hexagons):
    out = []
    for h in hexagons:
        for i in range(0, 6):
            pointx = round(h[0] + 2 * r * math.sin(math.pi / 3 * i + math.pi / 6), 10)
            pointy = round(h[1] + 2 * r * math.cos(math.pi / 3 * i + math.pi / 6), 10)
            if [pointx, pointy] not in hexagons:
                out.append([pointx, pointy])
    return out


hx = [0]
hy = [0]
hexagons = [[hx[0], hy[0]]]
n = 200
for i in range(1, n):
    neighbours = _hneighbours(hexagons)
    best_neighbour = neighbours[0]
    best_loss = _hloss(best_neighbour, polygon)
    for neighbour in neighbours:
        neighbour_loss = _hloss(neighbour, polygon)
        if neighbour_loss < best_loss:
            best_neighbour = neighbour
            best_loss = neighbour_loss
    hexagons.append(best_neighbour)
    print(i, best_neighbour)
    hx.append(best_neighbour[0])
    hy.append(best_neighbour[1])


# hexagons vs. map
fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=ys))
for h in hexagons:
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
        line=dict(
            color="red",
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
    width=700,
    height=500
)
fig.show()
fig.write_image(path + "images/cz_hexagons_200_example.png")


# squares
squares = [[0, 0]]
n = 200
for i in range(1, n):
    neighbours = _neighbours(squares)
    best_neighbour = neighbours[0]
    best_loss = _loss(best_neighbour, polygon)
    for neighbour in neighbours:
        neighbour_loss = _loss(neighbour, polygon)
        if neighbour_loss < best_loss:
            best_neighbour = neighbour
            best_loss = neighbour_loss
    squares.append(best_neighbour)
    print(i, best_neighbour)

sx = []
sy = []
for s in squares:
    sx.append(s[0])
    sy.append(s[1])

fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=ys))
for s in squares:
    fig.add_shape(
        type="rect",
        x0=s[0] - 0.5,
        y0=s[1] - 0.5,
        x1=s[0] + 0.5,
        y1=s[1] + 0.5,
        line=dict(
            color="red",
            width=2,
        )
    )
fig.add_trace(go.Scatter(
    x=sx,
    y=sy,
    text=list(range(0, n + 1)),
    mode="text",
    textposition="middle center"
))
fig.update_layout(
    autosize=False,
    width=700,
    height=500
)
fig.show()
fig.write_image(path + "images/cz_squares_200_example.png")
