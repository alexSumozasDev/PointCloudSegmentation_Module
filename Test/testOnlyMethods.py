import numpy as np
import matplotlib.pyplot as plt
from PCHSegmentation.PointCloudClasses import PointCloudGeneric
from PCHSegmentation.SegmentationAlgorithm import (computePointCloudSegments, computePointCloudSegmentsSlicePCA,
                                   computePointCloudSegmentsAllPCA, computePointCloudSegmentsMinMaxPCA)


def test_compute_segments_slice():

    num_points = int(input("Enter the number of points in the point cloud: "))
    slices_input = input("Enter the slices as a list of integers separated by commas (e.g., 3,3,3): ")
    slices = list(map(int, slices_input.split(",")))

    np.random.seed(42)
    point_cloud_data = np.random.rand(num_points, 3) * 100
    point_cloud = PointCloudGeneric(data=point_cloud_data)
    point_cloud.rotatePointCloud(np.asarray([20, 45, 0]))
    segments = computePointCloudSegmentsSlicePCA(point_cloud, slices)

    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        point_cloud_data[:, 0],
        point_cloud_data[:, 1],
        point_cloud_data[:, 2],
        c="gray",
        s=1,
    )
    ax1.set_title("Original Point Cloud")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(122, projection="3d")

    import matplotlib.colors as mcolors

    colors = list(mcolors.BASE_COLORS.values())[: len(segments)]
    while len(colors) < len(segments):
        colors += colors
    for idx, segment in enumerate(segments):
        points = np.asarray(segment.getPoints(normalized=False))
        ax2.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors[idx],
            s=5,
            label=f"Segment {idx}",
        )

    ax2.set_title("Segmented Point Cloud")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend()

    plt.show()


def test_compute_segments_allpca():

    num_points = int(input("Enter the number of points in the point cloud: "))

    np.random.seed(42)
    point_cloud_data = np.random.rand(num_points, 3) * 100
    point_cloud = PointCloudGeneric(data=point_cloud_data)

    segments = computePointCloudSegmentsAllPCA(point_cloud)

    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        point_cloud_data[:, 0],
        point_cloud_data[:, 1],
        point_cloud_data[:, 2],
        c="gray",
        s=1,
    )
    ax1.set_title("Original Point Cloud")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(122, projection="3d")

    import matplotlib.colors as mcolors

    colors = list(mcolors.BASE_COLORS.values())[: len(segments)]
    while len(colors) < len(segments):
        colors += colors
    for idx, segment in enumerate(segments):
        points = np.asarray(segment.getPoints(normalized=False))
        ax2.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors[idx],
            s=5,
            label=f"Segment {idx}",
        )

    ax2.set_title("Segmented Point Cloud")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend()

    plt.show()


def test_compute_segments_minpca():

    num_points = int(input("Enter the number of points in the point cloud: "))
    dir_input = input("Enter the direction: [Min or Max]")

    np.random.seed(42)
    point_cloud_data = np.random.rand(num_points, 3) * 100
    point_cloud = PointCloudGeneric(data=point_cloud_data)

    segments = computePointCloudSegmentsMinMaxPCA(point_cloud, dir_input)

    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        point_cloud_data[:, 0],
        point_cloud_data[:, 1],
        point_cloud_data[:, 2],
        c="gray",
        s=1,
    )
    ax1.set_title("Original Point Cloud")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(122, projection="3d")

    import matplotlib.colors as mcolors

    colors = list(mcolors.BASE_COLORS.values())[: len(segments)]
    while len(colors) < len(segments):
        colors += colors
    for idx, segment in enumerate(segments):
        points = np.asarray(segment.getPoints(normalized=False))
        ax2.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors[idx],
            s=5,
            label=f"Segment {idx}",
        )

    ax2.set_title("Segmented Point Cloud")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend()

    plt.show()


def testOnlyMethods():

    test_to_execute = input("Input method to test: [0 for AllPCA, 1 for MinPCA, 2 for SlicePCA]")

    if int(test_to_execute) == 0:
        test_compute_segments_allpca()
    elif int(test_to_execute) == 1:
        test_compute_segments_minpca()
    elif int(test_to_execute) == 2:
        test_compute_segments_slice()
    else:
        print("\n\tNot valid method selected.")


if __name__ == "__main__":

    testOnlyMethods()


