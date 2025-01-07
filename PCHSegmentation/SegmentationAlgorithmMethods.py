import numpy as np
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from PCHSegmentation.PointCloudClasses.PointCloudGeneric import PointCloudGeneric


def computePointCloudSegmentsAllPCA(point_cloud):
    """
    Segments a 3D point cloud into 8 regions (octants) based on the principal component analysis (PCA) axes.

    This function computes the PCA axes of the input point cloud, divides the point cloud into 8 regions
    (octants) using the PCA-transformed coordinates. The resulting segments are returned as separate `PointCloudGeneric` objects.

    Args:
        point_cloud (PointCloudGeneric):
            A 'PointCloudGeneric' object containing the 3D points to be segmented.

    Returns:
        list[PointCloudGeneric]:
            A list of 8 'PointCloudGeneric' objects, each corresponding to one octant. Each segment
            contains the points that fall within the respective octant in PCA space, and a unique
            color is assigned to distinguish it.
    """

    points = np.array(point_cloud.getPoints())

    if point_cloud.getPoints() is None or point_cloud is None:
        return [None, None, None, None, None, None, None, None]

    pca = PCA(n_components=3)
    pca.fit(points)
    object_axis = pca.components_

    point_cloud.object_axis = object_axis

    centroid = np.mean(points, axis=0)

    translated_points = points - centroid
    transformed_points = translated_points @ object_axis.T

    list_colors = [[1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0], [0.5, 1, 0], [0.3, 0.3, 0.3],[1, 0, 0]]

    octant_segments = []
    for i in range(2):
        for j in range(2):
            for k in range(2):

                mask = (
                    (transformed_points[:, 0] >= 0 if i == 0 else transformed_points[:, 0] < 0) &
                    (transformed_points[:, 1] >= 0 if j == 0 else transformed_points[:, 1] < 0) &
                    (transformed_points[:, 2] >= 0 if k == 0 else transformed_points[:, 2] < 0)
                )

                octant_points = points[mask]

                segment = PointCloudGeneric(data=octant_points.tolist(), n_id=i + j * 2 + k * 4)
                segment.color = list_colors[i + j * 2 + k * 4]

                octant_segments.append(segment)

    return octant_segments


def computePointCloudSegmentsMinMaxPCA(point_cloud, direction="Min"):
    """
    Computes octant segments of a point cloud using PCA, with the projection axis determined
    by the eigenvector corresponding to the smallest or largest eigenvalue.

    Args:
        point_cloud (PointCloudGeneric):
            A `PointCloudGeneric` object containing 3D points.
        direction (str):
            Determines the selection of the normal vector:
            - "Min": Use the eigenvector with the smallest eigenvalue.
            - "Max": Use the eigenvector with the largest eigenvalue.

    Returns:
        list[PointCloudGeneric]: A list of 8 segmented octants as `PointCloudGeneric` objects.
    """
    centroid = point_cloud.centroid
    points = point_cloud.getPoints(normalized=False)

    if len(points) < 5:
        return [None, None, None, None, None, None, None, None]

    pca = PCA(n_components=3)
    pca.fit(points)

    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_

    if direction.lower() == "min":
        normal_index = np.argmin(eigenvalues)
    elif direction.lower() == "max":
        normal_index = np.argmax(eigenvalues)
    else:
        raise ValueError("Invalid direction. Use 'Min' or 'Max'.")

    normal = eigenvectors[normal_index]

    def project_point_onto_plane(point, centroid, normal):
        vector_to_point = point - centroid
        distance_to_plane = np.dot(vector_to_point, normal)
        projection = point - distance_to_plane * normal
        return projection

    projected_points = np.array([project_point_onto_plane(p, centroid, normal) for p in points])

    arbitrary_vector = np.array([1, 0, 0]) if not np.allclose(normal, [1, 0, 0]) else np.array([0, 1, 0])

    u_axis = np.cross(normal, arbitrary_vector)
    u_axis /= np.linalg.norm(u_axis)

    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)

    projected_2d = np.array([[np.dot(p - centroid, u_axis), np.dot(p - centroid, v_axis)] for p in projected_points])

    hull = ConvexHull(projected_2d)

    def rotating_calipers_obb(hull_points):
        edges = np.diff(hull_points[np.append(hull.vertices, hull.vertices[0])], axis=0)
        edge_directions = edges / np.linalg.norm(edges, axis=1)[:, np.newaxis]
        orthogonal_directions = np.array([-edge_directions[:, 1], edge_directions[:, 0]]).T
        min_area = float('inf')
        best_rect = None
        for direction, orthogonal in zip(edge_directions, orthogonal_directions):
            projections = np.dot(hull_points, direction)
            orthogonal_projections = np.dot(hull_points, orthogonal)
            min_proj, max_proj = min(projections), max(projections)
            min_orth_proj, max_orth_proj = min(orthogonal_projections), max(orthogonal_projections)
            width = max_proj - min_proj
            height = max_orth_proj - min_orth_proj
            area = width * height
            if area < min_area:
                min_area = area
                best_rect = (direction, orthogonal, min_proj, max_proj, min_orth_proj, max_orth_proj)
        return best_rect, min_area

    best_rect, min_area = rotating_calipers_obb(projected_2d)
    direction, orthogonal, min_proj, max_proj, min_orth_proj, max_orth_proj = best_rect

    unit_vector_1_3d = direction[0] * u_axis + direction[1] * v_axis
    unit_vector_2_3d = orthogonal[0] * u_axis + orthogonal[1] * v_axis

    translated_points = points - centroid
    transformed_points = translated_points @ np.array([unit_vector_1_3d, unit_vector_2_3d, normal]).T

    list_colors = [[1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0], [0.5, 1, 0], [0.3, 0.3, 0.3], [1, 0, 0]]
    octant_segments = []

    for i in range(2):
        for j in range(2):
            for k in range(2):
                mask = (
                        (transformed_points[:, 0] >= 0 if i == 0 else transformed_points[:, 0] < 0) &
                        (transformed_points[:, 1] >= 0 if j == 0 else transformed_points[:, 1] < 0) &
                        (transformed_points[:, 2] >= 0 if k == 0 else transformed_points[:, 2] < 0)
                )

                octant_points = point_cloud.points[mask]

                segment = PointCloudGeneric(data=octant_points.tolist(), n_id=i + j * 2 + k * 4)
                segment.color = list_colors[i + j * 2 + k * 4]

                octant_segments.append(segment)

    return octant_segments


def computePointCloudSegmentsSlicePCA(point_cloud, slices=[2, 2, 2]):
    """
    Segments a 3D point cloud into regions along the PCA axes, based on a specified number of slices for each axis.

    Args:
        point_cloud (PointCloudGeneric):
            A `PointCloudGeneric` object containing the 3D points to segment.
        slices (list, tuple, or dict):
            A vector or dictionary specifying the number of slices for each PCA axis.
            - If list/tuple: Must have 3 values, e.g., [4, 3, 5].
            - If dict: Must have keys 'x', 'y', 'z', e.g., {'x': 4, 'y': 3, 'z': 5}.

    Returns:
        list[PointCloudGeneric]:
            A list of segmented regions as `PointCloudGeneric` objects.
    """

    points = np.asarray(point_cloud.getPoints())

    pca = PCA(n_components=3)
    pca_points = pca.fit_transform(points)

    if isinstance(slices, (list, tuple)) and len(slices) == 3:
        slice_counts = slices
    elif isinstance(slices, dict) and all(k in slices for k in ['x', 'y', 'z']):
        slice_counts = [slices['x'], slices['y'], slices['z']]
    else:
        raise ValueError("`slices` must be a list, tuple, or dictionary with 3 values.")

    pca_min = np.min(pca_points, axis=0)
    pca_max = np.max(pca_points, axis=0)
    bin_edges = [np.linspace(pca_min[i], pca_max[i], slice_counts[i] + 1) for i in range(3)]

    segment_labels = np.zeros(pca_points.shape[0], dtype=int)
    for i in range(3):
        axis_labels = np.digitize(pca_points[:, i], bin_edges[i]) - 1
        axis_labels = np.clip(axis_labels, 0, slice_counts[i] - 1)
        segment_labels += axis_labels * int(np.prod(slice_counts[:i]))

    segments = []
    total_segments = np.prod(slice_counts)

    for i in range(total_segments):
        segment_points = points[segment_labels == i]
        if len(segment_points) > 0:
            segment = PointCloudGeneric(data=segment_points.tolist(), n_id=i)
            segments.append(segment)

    return segments

