# This code is part of Grandalf
#  Copyright (C) 2011 Axel Tillequin (bdcht3@gmail.com) and others
# published under GPLv2 license or EPLv1 license
# Contributor(s): Axel Tillequin, Fabio Zadrozny

#  Edge routing algorithms.
#  These are mosty helpers for routing an edge 'e' through
#  points pts with various tweaks like moving the starting point
#  to the intersection with the bounding box and taking some constraints
#  into account, and/or moving the head also to its prefered position.
#  Of course, since gandalf only works with bounding boxes, the exact
#  shape of the nodes are not known and the edge drawing inside the bb
#  shall be performed by the drawing engine associated with 'views'.
#  (e.g. look at intersectC when the node shape is a circle)

from .utils.geometry import intersectR, getangle, sqrt

# ------------------------------------------------------------------------------
class EdgeViewer(object):
    def setpath(self, pts):
        self._pts = pts


# ------------------------------------------------------------------------------
#  basic edge routing with lines : nothing to do for routing
#  since the layout engine has already provided to list of points through which
#  the edge shall be drawn. We just compute the position where to adjust the
#  tail and head.
def route_with_lines(e, pts):
    assert hasattr(e, "view")
    tail_pos = intersectR(e.v[0].view, topt=pts[1])
    head_pos = intersectR(e.v[1].view, topt=pts[-2])
    pts[0] = tail_pos
    pts[-1] = head_pos
    e.view.head_angle = getangle(pts[-2], pts[-1])


# ------------------------------------------------------------------------------
#  enhanced edge routing where 'corners' of the above polyline route are
#  rounded with a bezier curve.
def route_with_splines(e, pts):
    from .utils.geometry import setroundcorner

    route_with_lines(e, pts)
    splines = setroundcorner(e, pts)
    e.view.splines = splines


def _gen_point(p1, p2, new_distance):
    from .utils.geometry import new_point_at_distance

    initial_distance = distance = sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    if initial_distance < 1e-10:
        return None
    if distance > new_distance:
        distance = distance - new_distance
    else:
        return None
    angle = getangle(p1, p2)
    new = new_point_at_distance(p1, distance, angle)
    return new


def _gen_smoother_middle_points_from_3_points(pts, initial):
    p1 = pts[0]
    p2 = pts[1]
    p3 = pts[2]
    distance1 = sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    distance2 = sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2)
    if distance1 < 1e-10 or distance2 < 1e-10:
        yield p2
    else:
        if distance1 < initial or distance2 < initial:
            yield p2
        else:
            p2a = _gen_point(p1, p2, initial)
            p2b = _gen_point(p3, p2, initial)
            if p2a is None or p2b is None:
                yield p2
            else:
                yield p2a
                yield p2b


# Future work: possibly work better when we already have 4 points?
# maybe: http://stackoverflow.com/questions/1251438/catmull-rom-splines-in-python
def _round_corners(pts, round_at_distance):
    if len(pts) > 2:
        calc_with_distance = round_at_distance
        while calc_with_distance > 0.5:
            new_lst = [pts[0]]
            for i, curr in enumerate(pts[1:-1]):
                i += 1
                p1 = pts[i - 1]
                p2 = curr
                p3 = pts[i + 1]
                if len(pts) > 3:
                    # i.e.: at least 4 points
                    if sqrt((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2) < (
                        2 * calc_with_distance
                    ):
                        # prevent from crossing over.
                        new_lst.append(p2)
                        continue
                generated = _gen_smoother_middle_points_from_3_points(
                    [p1, p2, p3], calc_with_distance
                )
                for j in generated:
                    new_lst.append(j)
            new_lst.append(pts[-1])
            pts = new_lst
            calc_with_distance /= 2.0
    return pts


# ------------------------------------------------------------------------------
# Routing with a custom algorithm to round corners
# It works by generating new points up to a distance from where an edge is
# found (and then iteratively refining based on that).
# This is a custom implementation as this interpolation method worked
# well for me where others weren't so great.

# This is the point where it'll start rounding from an edge.
# (can be changed to decide up to which distance it starts
# rounding from an edge).
ROUND_AT_DISTANCE = 40


def route_with_rounded_corners(e, pts):
    route_with_lines(e, pts)
    new_pts = _round_corners(pts, round_at_distance=ROUND_AT_DISTANCE)
    pts[:] = new_pts[:]
