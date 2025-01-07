from matplotlib import pyplot as plt
from PCHSegmentation.PointCloudClasses import PointCloudGeneric
import pandas as pd
from PCHSegmentation.SegmentationAlgorithm import computePointCloudSegments

p = r"C:\Users\asumo\Repos\Benchtool_PointCloudSegmentation\Scans\new_scans\20240730_UMcubo_FACE01_01\20240730_scanContinuous_sampling125_UMcubo_01.csv"


    #r"C:\Users\asumo\Repos\Benchtool_PointCloudSegmentation\Scans\scans\20231004_scan_10X10Y_20x20.csv"
    #r"C:\Users\asumo\Repos\Benchtool_PointCloudSegmentation\Scans\new_scans\20240730_UMcubo_FACE01_01\20240730_scanContinuous_sampling125_UMcubo_01.csv"
    #r"C:\Users\asumo\Repos\Benchtool_PointCloudSegmentation\Scans\new_scans\20240730_FORTUStriAsym_FACE01_01\20240730_scanContinuous_sampling125_FORTUStriAsym_03.csv"


def test_slice():
    ex_param = [{"cell_density_target": 500, "max_num_regions": 2000},
                {"cell_density_target": 100, "max_num_regions": 2000},
                {"cell_density_target": 20, "max_num_regions": 10000}]

    fig = plt.figure(figsize=(15, 5))

    columns = ["Date", "unknown_1", "x", "y", "z", "unknown_2", "unknown_3", "unknown_4", "unknown_5"]
    df = pd.read_csv(p, names=columns)
    df = df[["x", "y", "z"]]
    print(f"Tamaño de la nube de puntos {len(df)}")
    point_cloud = PointCloudGeneric(data=df.to_numpy())

    ax1 = fig.add_subplot(221, projection="3d")
    ax1.set_title(f"Nube de puntos escaneada")
    ax1.scatter(
        point_cloud.getPoints(normalized=False)[:, 0],
        point_cloud.getPoints(normalized=False)[:, 1],
        point_cloud.getPoints(normalized=False)[:, 2],
        color="#00FF00",
        s=1,
    )

    for i in range(3):
        ex_para = ex_param[i]

        database = computePointCloudSegments(point_cloud, method="SlicePCA", automatic=True, execution_parameters=ex_para, args={"num_slices": [1,1,1]})

        ax1 = fig.add_subplot(222 + i, projection="3d")

        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",  # Red, Green, Blue, Yellow, Magenta, Cyan
            "#FF4500", "#FFD700", "#8A2BE2", "#7FFF00", "#FF6347", "#40E0D0",
            "#32CD32", "#FF1493", "#00BFFF", "#8B0000", "#800080", "#FF8C00",
            "#A52A2A", "#5F9EA0", "#D2691E", "#9ACD32", "#F0E68C", "#ADFF2F",
            "#6495ED", "#FF7F50", "#9932CC", "#8B4513", "#2E8B57", "#DAA520",
            "#7CFC00", "#D2B48C", "#FFB6C1", "#8FBC8F", "#A9A9A9", "#006400",
            "#1E90FF", "#B22222", "#FF69B4", "#3CB371", "#FFFFF0", "#F08080",
            "#8A2BE2", "#A52A2A", "#C71585", "#8B008B", "#D2691E", "#7FFF00",
            "#FF1493", "#BDB76B", "#B0C4DE", "#556B2F", "#8B0000", "#B0E0E6",
            "#8FBC8F", "#F4A460", "#FFD700", "#FF00FF", "#7CFC00", "#32CD32"
        ]

        colors *= 100  # Extend color list to handle large numbers of segments

        segments = database # database.database[len(database.database) - 1]
        num_segments = len(segments)
        print(f"Segmentación {i + 1}: número de segmentos = {num_segments}")

        for n, seg in enumerate(segments):
            print(f"  Región {n + 1}: Color = {colors[n]}")
            ax1.scatter(
                seg.getPoints(normalized=False)[:, 0],
                seg.getPoints(normalized=False)[:, 1],
                seg.getPoints(normalized=False)[:, 2],
                color=colors[n],
                s=1,
            )
        ax1.set_title(f"Segmentación {i + 1}")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

    plt.show()


def test_rec():
    ex_param = [{"cell_density_target": 500, "max_num_regions": 2000},
                {"cell_density_target": 100, "max_num_regions": 2000},
                {"cell_density_target": 20, "max_num_regions": 10000}]

    fig = plt.figure(figsize=(15, 5))

    columns = ["Date", "unknown_1", "x", "y", "z", "unknown_2", "unknown_3", "unknown_4", "unknown_5"]
    df = pd.read_csv(p, names=columns)
    df = df[["x", "y", "z"]]

    point_cloud = PointCloudGeneric(data=df.to_numpy())

    ax1 = fig.add_subplot(221, projection="3d")

    ax1.set_title(f"Nube de puntos escaneada")

    ax1.scatter(
        point_cloud.getPoints(normalized=False)[:, 0],
        point_cloud.getPoints(normalized=False)[:, 1],
        point_cloud.getPoints(normalized=False)[:, 2],
        color="#00FF00",
        s=1,
    )


    for i in range(3):
        ex_para = ex_param[i]

        database = computePointCloudSegments(point_cloud, method="AllPCA", automatic=True,
                                             execution_parameters=ex_para, args={"num_slices": [1, 1, 1]})

        ax1 = fig.add_subplot(222 + i, projection="3d")

        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",  # Red, Green, Blue, Yellow, Magenta, Cyan
            "#FF4500", "#FFD700", "#8A2BE2", "#7FFF00", "#FF6347", "#40E0D0",
            "#32CD32", "#FF1493", "#00BFFF", "#8B0000", "#800080", "#FF8C00",
            "#A52A2A", "#5F9EA0", "#D2691E", "#9ACD32", "#F0E68C", "#ADFF2F",
            "#6495ED", "#FF7F50", "#9932CC", "#8B4513", "#2E8B57", "#DAA520",
            "#7CFC00", "#D2B48C", "#FFB6C1", "#8FBC8F", "#A9A9A9", "#006400",
            "#1E90FF", "#B22222", "#FF69B4", "#3CB371", "#FFFFF0", "#F08080",
            "#8A2BE2", "#A52A2A", "#C71585", "#8B008B", "#D2691E", "#7FFF00",
            "#FF1493", "#BDB76B", "#B0C4DE", "#556B2F", "#8B0000", "#B0E0E6",
            "#8FBC8F", "#F4A460", "#FFD700", "#FF00FF", "#7CFC00", "#32CD32"
        ]

        colors *= 100  # Extend color list to handle large numbers of segments

        segments = database.database[len(database.database) - 1]
        num_segments = len(segments)
        print(f"Experiment {i + 1}: Number of Segments = {num_segments}")

        for n, seg in enumerate(segments):
            print(f"  Segment {n + 1}: Color = {colors[n]}")
            ax1.scatter(
                seg.getPoints(normalized=False)[:, 0],
                seg.getPoints(normalized=False)[:, 1],
                seg.getPoints(normalized=False)[:, 2],
                color=colors[n],
                s=1,
            )
        ax1.set_title(f"Segmentación {i + 1}")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

    plt.show()

def test_recmin():
    ex_param = [{"cell_density_target": 500, "max_num_regions": 2000},
                {"cell_density_target": 100, "max_num_regions": 2000},
                {"cell_density_target": 20, "max_num_regions": 10000}]

    fig = plt.figure(figsize=(15, 5))

    columns = ["Date", "unknown_1", "x", "y", "z", "unknown_2", "unknown_3", "unknown_4", "unknown_5"]
    df = pd.read_csv(p, names=columns)
    df = df[["x", "y", "z"]]

    point_cloud = PointCloudGeneric(data=df.to_numpy())

    ax1 = fig.add_subplot(221, projection="3d")

    ax1.set_title(f"Nube de puntos escaneada")

    ax1.scatter(
        point_cloud.getPoints(normalized=False)[:, 0],
        point_cloud.getPoints(normalized=False)[:, 1],
        point_cloud.getPoints(normalized=False)[:, 2],
        color="#00FF00",
        s=1,
    )

    for i in range(3):
        ex_para = ex_param[i]

        database = computePointCloudSegments(point_cloud, method="MinPCA", automatic=True,
                                             execution_parameters=ex_para, args={"Direction": "Min"})

        ax1 = fig.add_subplot(222 + i, projection="3d")

        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",  # Red, Green, Blue, Yellow, Magenta, Cyan
            "#FF4500", "#FFD700", "#8A2BE2", "#7FFF00", "#FF6347", "#40E0D0",
            "#32CD32", "#FF1493", "#00BFFF", "#8B0000", "#800080", "#FF8C00",
            "#A52A2A", "#5F9EA0", "#D2691E", "#9ACD32", "#F0E68C", "#ADFF2F",
            "#6495ED", "#FF7F50", "#9932CC", "#8B4513", "#2E8B57", "#DAA520",
            "#7CFC00", "#D2B48C", "#FFB6C1", "#8FBC8F", "#A9A9A9", "#006400",
            "#1E90FF", "#B22222", "#FF69B4", "#3CB371", "#FFFFF0", "#F08080",
            "#8A2BE2", "#A52A2A", "#C71585", "#8B008B", "#D2691E", "#7FFF00",
            "#FF1493", "#BDB76B", "#B0C4DE", "#556B2F", "#8B0000", "#B0E0E6",
            "#8FBC8F", "#F4A460", "#FFD700", "#FF00FF", "#7CFC00", "#32CD32"
        ]

        colors *= 100  # Extend color list to handle large numbers of segments

        segments = database.database[len(database.database) - 1]
        num_segments = len(segments)
        print(f"Experiment {i + 1}: Number of Segments = {num_segments}")

        for n, seg in enumerate(segments):
            print(f"  Segment {n + 1}: Color = {colors[n]}")
            ax1.scatter(
                seg.getPoints(normalized=False)[:, 0],
                seg.getPoints(normalized=False)[:, 1],
                seg.getPoints(normalized=False)[:, 2],
                color=colors[n],
                s=1,
            )
        ax1.set_title(f"Segmentación {i + 1}")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

    plt.show()


test_slice()
test_rec()
test_recmin()