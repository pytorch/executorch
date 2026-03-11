#!/usr/bin/env python
#
# This code is part of Grandalf
# Copyright (C) 2008 Axel Tillequin (bdcht3@gmail.com) and others
# published under GPLv2 license or EPLv1 license
# Contributor(s): Axel Tillequin, Fabio Zadrozny
from .poset import *
from .dot import *

from math import atan2, sqrt
from random import SystemRandom

# ------------------------------------------------------------------------------
def intersect2lines(xy1, xy2, xy3, xy4):
    (x1, y1) = xy1
    (x2, y2) = xy2
    (x3, y3) = xy3
    (x4, y4) = xy4
    b = (x2 - x1, y2 - y1)
    d = (x4 - x3, y4 - y3)
    det = b[0] * d[1] - b[1] * d[0]
    if det == 0:
        return None
    c = (x3 - x1, y3 - y1)
    t = float(c[0] * b[1] - c[1] * b[0]) / (det * 1.0)
    if t < 0.0 or t > 1.0:
        return None
    t = float(c[0] * d[1] - c[1] * d[0]) / (det * 1.0)
    if t < 0.0 or t > 1.0:
        return None
    x = x1 + t * b[0]
    y = y1 + t * b[1]
    return (x, y)


# ------------------------------------------------------------------------------
#  intersectR returns the intersection point between the Rectangle
#  (w,h) that characterize the view object and the line that goes
#  from the views' object center to the 'topt' point.
def intersectR(view, topt):
    # we compute intersection in local views' coord:
    # center of view is obviously :
    x1, y1 = 0, 0
    # endpoint in view's coord:
    x2, y2 = topt[0] - view.xy[0], topt[1] - view.xy[1]
    # bounding box:
    bbx2 = view.w // 2
    bbx1 = -bbx2
    bby2 = view.h // 2
    bby1 = -bby2
    # all 4 segments of the bb:
    S = [
        ((x1, y1), (x2, y2), (bbx1, bby1), (bbx2, bby1)),
        ((x1, y1), (x2, y2), (bbx2, bby1), (bbx2, bby2)),
        ((x1, y1), (x2, y2), (bbx1, bby2), (bbx2, bby2)),
        ((x1, y1), (x2, y2), (bbx1, bby2), (bbx1, bby1)),
    ]
    # check intersection with each seg:
    for segs in S:
        xy = intersect2lines(*segs)
        if xy != None:
            x, y = xy
            # return global coord:
            x += view.xy[0]
            y += view.xy[1]
            return (x, y)
    # there can't be no intersection unless the endpoint was
    # inside the bb !
    raise ValueError(
        "no intersection found (point inside ?!). view: %s topt: %s" % (view, topt)
    )


# ------------------------------------------------------------------------------
def getangle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    theta = atan2(y2 - y1, x2 - x1)
    return theta


# ------------------------------------------------------------------------------
def median_wh(views):
    mw = [v.w for v in views]
    mh = [v.h for v in views]
    mw.sort()
    mh.sort()
    return (mw[len(mw) // 2], mh[len(mh) // 2])


# ------------------------------------------------------------------------------
try:
    from numpy import array, matrix, cos, sin

    has_numpy = True
except ImportError:
    has_numpy = False
    from math import cos, sin, pi
    from .linalg import array, matrix

#  rand_ortho1 returns a numpy.array representing
#  a random normalized n-dimension vector orthogonal to (1,1,1,...,1).
def rand_ortho1(n):
    r = SystemRandom()
    pos = [r.random() for x in range(n)]
    s = sum(pos)
    v = array(pos, dtype=float) - (s / float(n))
    norm = sqrt(sum(v * v))
    return v / norm


# ------------------------------------------------------------------------------
#  intersectC returns the intersection point between the Circle
#  of radius r and centered on views' position with the line
#  to the 'topt' point.
def intersectC(view, r, topt):
    theta = getangle(view.xy, topt)
    x = int(cos(theta) * r)
    y = int(sin(theta) * r)
    return (x, y)


# ------------------------------------------------------------------------------
#  setcurve returns the spline curve that path through the list of points P.
#  The spline curve is a list of cubic bezier curves (nurbs) that have
#  matching tangents at their extreme points.
#  The method considered here is taken from "The NURBS book" (Les A. Piegl,
#  Wayne Tiller, Springer, 1997) and implements a local interpolation rather
#  than a global interpolation.
def setcurve(e, pts, tgs=None):
    P = list(map(array, pts))
    n = len(P)
    # tangent estimation
    if tgs:
        assert len(tgs) == n
        T = list(map(array, tgs))
        Q = [P[k + 1] - P[k] for k in range(0, n - 1)]
    else:
        Q, T = tangents(P, n)
    splines = []
    for k in range(n - 1):
        t = T[k] + T[k + 1]
        a = 16.0 - (t.dot(t))
        b = 12.0 * (Q[k].dot(t))
        c = -36.0 * Q[k].dot(Q[k])
        D = (b * b) - 4.0 * a * c
        assert D >= 0
        sd = sqrt(D)
        s1, s2 = (-b - sd) / (2.0 * a), (-b + sd) / (2.0 * a)
        s = s2
        if s1 >= 0:
            s = s1
        C0 = tuple(P[k])
        C1 = tuple(P[k] + (s / 3.0) * T[k])
        C2 = tuple(P[k + 1] - (s / 3.0) * T[k + 1])
        C3 = tuple(P[k + 1])
        splines.append([C0, C1, C2, C3])
    return splines


# ------------------------------------------------------------------------------
def tangents(P, n):
    assert n >= 2
    Q = []
    T = []
    for k in range(0, n - 1):
        q = P[k + 1] - P[k]
        t = q / sqrt(q.dot(q))
        Q.append(q)
        T.append(t)
    T.append(t)
    return (Q, T)


# ------------------------------------------------------------------------------
def setroundcorner(e, pts):
    P = list(map(array, pts))
    n = len(P)
    Q, T = tangents(P, n)
    c0 = P[0]
    t0 = T[0]
    k0 = 0
    splines = []
    k = 1
    while k < n:
        z = abs(t0[0] * T[k][1] - (t0[1] * T[k][0]))
        if z < 1.0e-6:
            k += 1
            continue
        if (k - 1) > k0:
            splines.append([c0, P[k - 1]])
        if (k + 1) < n:
            splines.extend(setcurve(e, [P[k - 1], P[k + 1]], tgs=[T[k - 1], T[k + 1]]))
        else:
            splines.extend(setcurve(e, [P[k - 1], P[k]], tgs=[T[k - 1], T[k]]))
            break
        if (k + 2) < n:
            c0 = P[k + 1]
            t0 = T[k + 1]
            k0 = k + 1
            k += 2
        else:
            break
    return splines or [[P[0], P[-1]]]


# ------------------------------------------------------------------------------
def new_point_at_distance(pt, distance, angle):
    # angle in radians
    distance = float(distance)
    x, y = pt[0], pt[1]
    x += distance * cos(angle)
    y += distance * sin(angle)
    return x, y
