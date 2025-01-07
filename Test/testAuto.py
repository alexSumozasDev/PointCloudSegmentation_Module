
from PCHSegmentation.SegmentationAlgorithm import applySegmentationSlice
from matplotlib.animation import FuncAnimation
from PCHSegmentation.PointCloudClasses import PointCloudGeneric
from PCHSegmentation.SegmentationAlgorithm import applyRecursiveSegmentation, computePointCloudSegments
from matplotlib import pyplot as plt
import numpy as np
from PCHSegmentation.DataBase import PointCloudDatabase


def test_apply_segmentation_slice():

    num_points = int(input("Enter the number of points in the point cloud: "))
    cell_density_target = int(input("Enter the target cell density: "))
    max_cells = int(input("Enter the max number of cells: "))

    np.random.seed(42)
    point_cloud_data = np.random.rand(num_points, 3) * 100
    point_cloud = PointCloudGeneric(data=point_cloud_data)

    segments = applySegmentationSlice(point_cloud, cell_density_target, max_cells)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(point_cloud_data[:, 0], point_cloud_data[:, 1], point_cloud_data[:, 2], c='gray', s=1)
    ax1.set_title('Original Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(122, projection='3d')

    import matplotlib.colors as mcolors
    colors = list(mcolors.BASE_COLORS.values())[:len(segments)]
    while len(colors) < len(segments):
        colors += colors

    for idx, segment in enumerate(segments):
        points = np.asarray(segment.getPoints())
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors[idx], s=5, label=f'Segment {idx + 1}')

    ax2.set_title('Segmented Point Cloud')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()

    plt.show()


def test_applyRecursiveSegmentation():
    total_points = int(input("Enter number of Points Point CLoud: "))
    method = input("Enter Metod 'AllPCA' or 'MinPCA'")
    cell_density_target = int(input("Enter cell density target: "))
    max_num_regions = int(input("Enter max number of cells: "))
    min_cell_density = int(input("Enter min cell density: "))

    if method.upper() == "MINPCA":
        direc = input("Enter direction: 'Min' , 'Max'")
        args = {"Direction": direc}

    high_density_points = int(total_points * 0.95)
    high_density_region = np.random.uniform(
        low=[0.0, 0.0, 0.0],
        high=[0.6, 1.0, 1.0],
        size=(high_density_points, 3)
    )

    low_density_points = int(total_points * 0.05)
    low_density_region = np.random.uniform(
        low=[0.4, 0.0, 0.0],
        high=[1.0, 1.0, 1.0],
        size=(low_density_points, 3)
    )

    point_cloud_data = np.vstack((high_density_region, low_density_region))

    point_cloud = PointCloudGeneric(point_cloud_data)

    args = {"Direction": "Min"}

    segmented_db = applyRecursiveSegmentation(
        point_cloud, method, cell_density_target, max_num_regions, min_cell_density, args
    )

    plot_segmented_point_cloud(segmented_db)


def plot_segmented_point_cloud(point_cloud_db):

    all_segments = [segment for segment in point_cloud_db.database[1]]

    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",  # Red, Green, Blue, Yellow, Magenta, Cyan
        "#FF4500", "#FFD700", "#8A2BE2", "#7FFF00", "#FF6347", "#40E0D0",
        # OrangeRed, Gold, BlueViolet, Chartreuse, Tomato, Turquoise
        "#32CD32", "#FF1493", "#00BFFF", "#8B0000", "#800080", "#FF8C00",
        # LimeGreen, DeepPink, DeepSkyBlue, DarkRed, Purple, DarkOrange
        "#A52A2A", "#5F9EA0", "#D2691E", "#9ACD32", "#F0E68C", "#ADFF2F",
        # Brown, CadetBlue, Chocolate, YellowGreen, Khaki, GreenYellow
        "#6495ED", "#FF7F50", "#9932CC", "#8B4513", "#2E8B57", "#DAA520",
        # CornflowerBlue, Coral, DarkOrchid, SaddleBrown, SeaGreen, Goldenrod
        "#7CFC00", "#D2B48C", "#FFB6C1", "#8FBC8F", "#A9A9A9", "#006400",
        # LawnGreen, Tan, LightPink, DarkSeaGreen, DarkGray, DarkGreen
        "#1E90FF", "#B22222", "#FF69B4", "#3CB371", "#FFFFF0", "#F08080",
        # DodgerBlue, Firebrick, HotPink, MediumSeaGreen, Ivory, LightCoral
        "#8A2BE2", "#A52A2A", "#C71585", "#8B008B", "#D2691E", "#7FFF00",
        # BlueViolet, Brown, MediumVioletRed, DarkMagenta, Chocolate, Chartreuse
        "#FF1493", "#BDB76B", "#B0C4DE", "#556B2F", "#8B0000", "#B0E0E6",
        # DeepPink, DarkKhaki, LightSteelBlue, DarkOliveGreen, DarkRed, PowderBlue
        "#8FBC8F", "#F4A460", "#FFD700", "#FF00FF", "#7CFC00", "#32CD32"
        # DarkSeaGreen, SandyBrown, Gold, Magenta, LawnGreen, LimeGreen
    ]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter([], [], [], color=[], alpha=0.6)

    def update(frame):
        ax.cla()
        ax.set_title("Segmented Point Cloud")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        for i in range(frame + 1):
            pts = all_segments[i].getPoints(normalized=False)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[i % len(colors)], alpha=0.6)

        return scatter,

    ani = FuncAnimation(fig, update, frames=len(all_segments), interval=200, repeat=False)

    plt.show()


def testCompleteAlgorithm():
    total_points = int(input("Enter the number of points in the point cloud: "))
    cell_density_target = int(input("Enter cell density target: "))
    max_num_regions = int(input("Enter max number of regions: "))
    max_cell_density = int(input("Enter max cell density: "))

    high_density_points = int(total_points * 0.95)
    high_density_region = np.random.uniform(
        low=[0.0, 0.0, 0.0],
        high=[0.6, 1.0, 1.0],
        size=(high_density_points, 3)
    )

    low_density_points = int(total_points * 0.05)
    low_density_region = np.random.uniform(
        low=[0.4, 0.0, 0.0],
        high=[1.0, 1.0, 1.0],
        size=(low_density_points, 3)
    )

    point_cloud_data = np.vstack((high_density_region, low_density_region))
    point_cloud = PointCloudGeneric(point_cloud_data)

    methods = ["AllPCA", "MinPCA", "MaxPCA", "SlicePCA"]
    args = {"Direction": "Min"}

    execution_parameters = {
        "cell_density_target": cell_density_target,
        "max_num_regions": max_num_regions,
        "min_cell_density": max_cell_density,
        "ignore_empty_cell": False
    }

    for method in methods:
        print(f"\nTesting method: {method}")

        if method == "SlicePCA":
            execution_parameters["slices"] = [2, 2, 2]

        segments = computePointCloudSegments(
            point_cloud=point_cloud,
            method=method,
            automatic=True,
            execution_parameters=execution_parameters,
            args=args,
            show_info=True
        )
        a = segments

        print(f"Finished testing method: {method}")


def testAuto():
    selec = input("Introduce method: 0 allpca, 1 minpca, 2 slicePCA, 3 ")

    if selec == "0" or selec == "1":

        test_applyRecursiveSegmentation()
    elif selec == "3":
        testCompleteAlgorithm()
    else:
        test_apply_segmentation_slice()


if __name__ == "__main__":
    testAuto()
