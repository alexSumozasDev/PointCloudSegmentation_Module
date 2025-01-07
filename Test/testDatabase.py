from PCHSegmentation.DataBase.PointCloudDatabase import PointCloudDatabase, AdaptivePointCloudDatabase
import numpy as np
from PCHSegmentation.PointCloudClasses.PointCloudGeneric import PointCloudGeneric
from PCHSegmentation.SegmentationAlgorithmMethods import (computePointCloudSegmentsAllPCA, computePointCloudSegmentsMinMaxPCA,
                                          computePointCloudSegmentsSlicePCA)
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


def testDatabase():
    num_points = int(input("Enter the number of points of the point cloud: "))
    num_div = int(input("Enter the number of levels of subdivision: "))
    method_input = int(input("Enter the method: [0 for AllPCA, 1 for MinPCA, 2 for SlicePCA] "))

    if method_input == 1:
        dir_used = input("Enter direction [Min, Max]: ")

    if method_input == 2:
        slices_input = input("Enter the slices as a list of integers separated by commas (e.g., 3,3,3): ")
        slices = list(map(int, slices_input.split(",")))

    highlight_lev = int(input("Enter a level to highlight: "))
    highlight_seg = int(input("Enter a seg to highlight [if input -1 all seg of level will be highlighted]: "))

    np.random.seed(42)
    point_cloud_data = np.random.rand(num_points, 3) * 100
    point_cloud = PointCloudGeneric(data=point_cloud_data)

    pc_database = PointCloudDatabase(point_cloud)

    counter = 0

    if method_input == 0:
        method_used = computePointCloudSegmentsAllPCA
        dict_used = {}
    elif method_input == 1:
        method_used = computePointCloudSegmentsMinMaxPCA
        dict_used = {"Direction": dir_used}
    else:
        method_used = computePointCloudSegmentsSlicePCA
        dict_used = {"Slices": slices}

    while counter < num_div:
        pc_database.subdivideLevel(method_used, dict_used, counter)

        counter += 1

    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(131, projection="3d")
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

    ax2 = fig.add_subplot(132, projection="3d")

    segments = pc_database.database[num_div]

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

    ax3 = fig.add_subplot(133, projection="3d")

    if highlight_seg == -1:
        segments = pc_database.database[highlight_lev]
    else:
        segments = [pc_database.database[highlight_lev][highlight_seg]]

    while len(colors) < len(segments):
        colors += colors
    for idx, segment in enumerate(segments):
        points = np.asarray(segment.getPoints(normalized=False))
        ax3.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=colors[highlight_seg],
            s=5,
            label=f"Highlight level: {highlight_lev} seg: {highlight_seg}",
        )

    ax3.set_title("Segmented Point Cloud")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.legend()

    plt.show()


def testDatabaseAdaptative():

    cell_density_min = 100
    num_points = int(input("Enter the number of points of the point cloud: "))
    num_div = int(input("Enter the number of levels of subdivision: "))
    method_input = int(input("Enter the method: [0 for AllPCA, 1 for MinPCA, 2 for SlicePCA] "))

    if method_input == 1:
        dir_used = input("Enter direction [Min, Max]: ")

    if method_input == 2:
        slices_input = input("Enter the slices as a list of integers separated by commas (e.g., 3,3,3): ")
        slices = list(map(int, slices_input.split(",")))

    highlight_lev = int(input("Enter a level to highlight: "))
    highlight_seg = int(input("Enter a seg to highlight [if input -1 all seg of level will be highlighted]: "))

    total_points = 50000

    cube_size = 1.0

    high_density_points = int(total_points * 0.99)
    high_density_region = np.random.uniform(
        low=[0.0, 0.0, 0.0],
        high=[0.5, 1.0, 1.0],
        size=(high_density_points, 3)
    )

    low_density_points = int(total_points * 0.01)
    low_density_region = np.random.uniform(
        low=[0.5, 0.0, 0.0],
        high=[1.0, 1.0, 1.0],
        size=(low_density_points, 3)
    )

    point_cloud_data = np.vstack((high_density_region, low_density_region))

    point_cloud = PointCloudGeneric(data=point_cloud_data)

    pc_database = AdaptivePointCloudDatabase(point_cloud)

    counter = 0

    if method_input == 0:
        method_used = computePointCloudSegmentsAllPCA
        dict_used = {}
    elif method_input == 1:
        method_used = computePointCloudSegmentsMinMaxPCA
        dict_used = {"Direction": dir_used}
    else:
        method_used = computePointCloudSegmentsSlicePCA
        dict_used = {"Slices": slices}

    while counter < num_div:
        pc_database.subdivideLevel(method_used, dict_used, counter, min_cell_density=cell_density_min)

        counter += 1

    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(131, projection="3d")
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

    ax2 = fig.add_subplot(132, projection="3d")

    segments = pc_database.database[num_div]

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

    ax3 = fig.add_subplot(133, projection="3d")

    if highlight_seg == -1:
        segments = pc_database.database[highlight_lev]
    else:
        segments = [pc_database.database[highlight_lev][highlight_seg]]

    while len(colors) < len(segments):
        colors += colors
    for idx, segment in enumerate(segments):
        points = np.asarray(segment.getPoints(normalized=False))
        ax3.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=colors[highlight_seg],
            s=5,
            label=f"Highlight level: {highlight_lev} seg: {highlight_seg}",
        )

    ax3.set_title("Segmented Point Cloud")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.legend()

    plt.show()


if __name__ == "__main__":

    test_to_execute = input("Enter database type to test [Standard, Adaptative]")

    if test_to_execute == "Adaptative":
        testDatabaseAdaptative()
    else:
        testDatabase()