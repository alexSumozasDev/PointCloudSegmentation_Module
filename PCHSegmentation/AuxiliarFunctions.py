import numpy as np
from random import random


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2, units="Degrees"):
    """ Returns the angle in radians between vectors 'v1' and 'v2' """

    factor_conversion = 180/np.pi if units == "Degrees" else 1
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * factor_conversion


def project_to_plane(points, centroid, generalAxis):

    points = np.array(points)
    centroid = np.array(centroid)
    generalAxis = np.array(generalAxis)

    normal = generalAxis / np.linalg.norm(generalAxis)

    vectors = points - centroid

    projection_lengths = np.dot(vectors, normal)

    projected_points = points - np.outer(projection_lengths, normal)

    return projected_points


def project_to_plane2(points, centroid, normal):
    normal = normal / np.linalg.norm(normal)
    distances = np.dot(points - centroid, normal)
    projected_points = points - np.outer(distances, normal)

    return projected_points[:, :2], projected_points


def compute_average_points(segments):
    return sum(len(segment.points) for segment in segments) / len(segments) if len(segments) > 0 else 0
