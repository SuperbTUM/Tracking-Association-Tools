import cv2
from math import acos, cos, sin
import numpy as np

# Radius of earth
R = 6371000


def distance_gps(longitude1, latitude1, longitude2, latitude2):
    d = R * acos(cos(latitude2) * cos(latitude1) * cos(longitude2 - longitude1) + sin(latitude2) * sin(latitude1))
    return d


def coordinate2gps(boundary_coordinate, boundary_gps, coordinate):
    x_relative = coordinate[0] / boundary_coordinate[0]
    y_relative = coordinate[1] / boundary_coordinate[1]
    # topleft, topright, bottomleft, bottomright
    lat = boundary_gps[0][0] + (boundary_gps[2][0] - boundary_gps[0][0]) * y_relative
    lon = boundary_gps[0][1] + (boundary_gps[1][1] - boundary_gps[0][1]) * x_relative
    return lat, lon


def calibration(src_points, dst_points, src, dst, points):
    h_src, w_src = src.shape[:-1]
    h_dst, w_dst = dst.shape[:-1]
    M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    # M = cv2.perspectiveTransform(src_points, dst_points)
    # pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    # dst = cv2.perspectiveTransform(pts, M)
    # dst = cv2.warpPerspective(src, M, (w_dst, h_dst))
    transformed = cv2.perspectiveTransform(points.reshape(-1, 1, 2), M)
    return transformed
