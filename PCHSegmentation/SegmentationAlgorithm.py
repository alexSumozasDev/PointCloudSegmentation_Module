import numpy as np
from sklearn.decomposition import PCA

from PCHSegmentation.SegmentationAlgorithmMethods import (
    computePointCloudSegmentsAllPCA, computePointCloudSegmentsMinMaxPCA,
    computePointCloudSegmentsSlicePCA)

from PCHSegmentation.DataBase.PointCloudDatabase import PointCloudDatabase
from PCHSegmentation.AuxiliarFunctions import compute_average_points


def computePointCloudSegments(point_cloud, method="AllPCA", show_info=False, automatic=False, execution_parameters=None, args={}):
    """
    Segment the point cloud based on the methods AllPCA, SlicePCA, MinPCA, MaxPCA.

    If 'automatic' is True, use the parameters in 'execution_parameters' to automatically
    compute the segmentation. Otherwise, the method is chosen manually via the 'method' parameter.

    :param point_cloud: The PointCloudGeneric object containing the 3D points to be segmented.
    :param method: Segmentation method to apply. Supported values are:
                   "MaxPCA", "MinPCA", "AllPCA", "SlicePCA".
    :param show_info: Flag to show the detailed info.

    :param automatic: If True, the segmentation will be done automatically based on 'execution_parameters'.
    :param execution_parameters: Dictionary containing parameters for automatic segmentation (only used if `automatic=True`).
                                 Posible keys: 'cell_density_target', 'max_num_regions', 'max_cell_density', 'ignore_empty_cell'.
    :param args: Arguments for segmentation methods (e.g., number of slices, direction).
    :return: List of PointCloudGeneric objects representing the segmented point clouds, or a PointCloudDatabase if automatic
             and methods AllPCA or MinPCA.
    """
    if point_cloud.getPoints() is None or point_cloud is None:
        return [None, None, None, None, None, None, None, None]

    if automatic:
        if execution_parameters is None:
            raise ValueError("Execution parameters must be provided when 'automatic' is True.")

        segments = computeAutomaticSegmentation(point_cloud, execution_parameters, method, args, show_details=False)

        return segments

    else:

        if method.upper() == "MAXPCA":
            segments = computePointCloudSegmentsMinMaxPCA(point_cloud, direction="Max")
        elif method.upper() == "MINPCA":
            segments = computePointCloudSegmentsMinMaxPCA(point_cloud, direction="Min")
        elif method.upper() == "ALLPCA":
            segments = computePointCloudSegmentsAllPCA(point_cloud)
        elif method.upper() == "SLICEPCA":
            slices = execution_parameters.get('slices', 3)
            segments = computePointCloudSegmentsSlicePCA(point_cloud, slices)

        if show_info:
            for idx, segment in enumerate(segments):
                segment_points = np.array(segment.points)
                centroid = np.mean(segment_points, axis=0)
                pca = PCA(n_components=3)
                pca.fit(segment_points)

                print(f"\nSegment {idx + 1}:")
                print(f"  Centroid: {centroid}")
                print(f"  Axis (PCA components): {pca.components_}")
                print(f"  Number of points in segment: {len(segment_points)}")

        return segments


def computeAutomaticSegmentation(point_cloud, execution_parameters, method, args, show_details=False):
    """
    Perform automatic segmentation based on provided parameters.

    :param execution_parameters:
    :param point_cloud: The PointCloudGeneric object containing the 3D points to be segmented.
    :param cell_density_target: Target cell density for segmentation.
    :param max_num_regions: Maximum number of regions to be created.
    :param max_cell_density: Maximum allowed cell density for segmentation.
    :param ignore_empty_cell: Flag to ignore empty cells when segmenting.
    :param method: The segmentation method being used (e.g., "SLICEPCA", "ALLPCA", etc.)
    :return: List of PointCloudGeneric objects representing the segmented point clouds.
    """

    print("Executing automatic segmentation with parameters:")

    cell_density_target = execution_parameters.get('cell_density_target', 50)
    max_num_regions = execution_parameters.get('max_num_regions', 70)
    max_cell_density = execution_parameters.get('max_cell_density', 999999)
    ignore_empty_cell = execution_parameters.get('ignore_empty_cell', False)

    print(f"cell_density_target={cell_density_target}, max_num_regions={max_num_regions}, "
          f"max_cell_density={max_cell_density}, ignore_empty_cell={ignore_empty_cell}, method={method}")

    if method.upper() == "SLICEPCA":

        print("Applying automatic segmentation for SlicePCA...")

        in_slices = args.get('num_slices', [2, 2, 2])

        segments = applySegmentationSlice(point_cloud, cell_density_target, max_number_cells=max_num_regions, slices=in_slices)


    else:
        print("Applying automatic segmentation for PCA-based methods...")

        Dir = args.get("Direction", "Min")

        segments = applyRecursiveSegmentation(point_cloud, method, cell_density_target, max_num_regions,
                                              max_cell_density, Direction=Dir)

    return segments


def applyRecursiveSegmentation(point_cloud, method, cell_density_target, max_num_regions,
                               max_cell_density, Direction="Min"):
    """
    Apply recursive segmentation until conditions are met and store results in a PointCloudDatabase.

    :param point_cloud: The PointCloudGeneric object containing the 3D points to be segmented.
    :param method: The segmentation method ("AllPCA", "MinPCA", or "MaxPCA").
    :param cell_density_target: Target cell density for segmentation.
    :param max_num_regions: Maximum number of regions to be created.
    :param max_cell_density: Maximum allowed cell density for segmentation.
    :return: The PointCloudDatabase instance containing the segmented point clouds.
    """

    point_cloud_db = PointCloudDatabase(point_cloud)

    level = 0
    segments = point_cloud_db.database[level]

    if method.upper() == "ALLPCA":
        method_used = computePointCloudSegmentsAllPCA

    elif method.upper() in ["MAXPCA", "MINPCA"]:
        method_used = computePointCloudSegmentsMinMaxPCA

    while not checkSegmentationConditions(segments, cell_density_target, max_num_regions, max_cell_density):
        print(f"Starting segmentation for level {level}. Total segments: {len(segments)}")

        total_points_before = sum(len(segment.points) for segment in segments) if segments else len(point_cloud.getPoints())
        print(f"  Total points before segmentation: {total_points_before}")

        point_cloud_db.subdivideLevel(method_used, Direction, level)

        level+=1

        segments = point_cloud_db.database[level]



        total_points_after = sum(len(segment.points) for segment in segments)
        print(f"  Total points after segmentation: {total_points_after}")
        if total_points_after != total_points_before:
            print(f"  Warning: Point mismatch detected! Before: {total_points_before}, After: {total_points_after}")

    return point_cloud_db


def checkSegmentationConditions(segments, cell_density_target, max_num_regions, max_cell_density):
    """
    Check if the segmentation conditions are met.

    :param segments: List of segments to check.
    :param cell_density_target: Target number of points per segment.
    :param max_num_regions: Maximum number of allowed segments.
    :param max_cell_density: Maximum allowed points in a segment.
    :return: True if conditions are met, otherwise False.
    """
    total_points = sum(len(segment.points) for segment in segments)
    average_points = total_points / len(segments) if len(segments) > 0 else 0

    print(f"Average points {average_points} Nregions {len(segments)} other {abs(average_points - cell_density_target) > cell_density_target * 0.1}")
    cond = False

    if average_points < cell_density_target: # (average_points - cell_density_target) < cell_density_target * 0.1:
        cond = True
    if any(len(segment.points) > max_cell_density for segment in segments):
        cond = False
    if len(segments) > max_num_regions:
        cond = True

    return cond


def applySegmentationSlice(point_cloud, cell_density_target, max_number_cells=99999, slices=[1,1,1]):
    """
    Segments the point cloud using SlicePCA segmentation and adjusts the slicing until conditions are met.

    :param point_cloud: The PointCloudGeneric object containing the 3D points to be segmented.
    :param cell_density_target: The target cell density (average number of points per segment).
    :return: A list of segmented PointCloudGeneric objects.
    """

    segments = computePointCloudSegmentsSlicePCA(point_cloud, slices)
    upper_limit_factor = 1.2
    lower_limit_factor = 0.8

    def adjust_slices(average_points, slices):

        diff = (average_points - cell_density_target)

        if diff > cell_density_target * upper_limit_factor:
            slices = [s + 1 for s in slices]

        elif diff > cell_density_target * lower_limit_factor:
            slices[0] += 1
            slices[1] += 1

        else:
            slices[0] += 1
        return slices

    while True:
        if len(segments) > max_number_cells: break

        average_points = compute_average_points(segments)

        print(f"Average points: {average_points}, Target: {cell_density_target}")

        if (average_points - cell_density_target) <= cell_density_target * 0.1:
            break

        slices = adjust_slices(average_points, slices)

        segments = computePointCloudSegmentsSlicePCA(point_cloud, slices)

        print(f"Adjusting slices to {slices}. Average points per segment: {average_points}")

    print(f"Segmentation complete. Average points per segment: {average_points}. Target: {cell_density_target}")

    return segments
