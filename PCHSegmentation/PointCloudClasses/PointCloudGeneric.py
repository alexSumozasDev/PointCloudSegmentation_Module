import numpy as np

class PointCloudGeneric:
    def __init__(self, data=None, n_id=0, color="#DC965A", p_ids=None):
        """
        Make a Point Cloud Object
        :param data: Can be a NumPy array of shape (N, 3) or a list of lists [[x1, y1, z1], [x2, y2, z2], ...]
        :param n_id: ID of the PointCloud
        :param color: Color in HEX or RGBA (tuple/list with 4 elements)
        """
        self.data_format = None
        self.object_axis = None
        self.color = self.parseColor(color)
        self.point_cloud_id = n_id
        self.normalized_points = None

        if data is not None:
            if isinstance(data, np.ndarray):
                if data.shape[1] == 3:
                    self.points = np.hstack([data, np.arange(len(data)).reshape(-1, 1)])
                    self.data_format = "numpy_array"
                elif data.shape[1] == 4:
                    self.points = data
                    self.data_format = "numpy_array"
                else:
                    raise ValueError("Data must have 3 or 4 columns (x, y, z, [point_id]).")

            elif isinstance(data, list) and all(isinstance(item, list) and len(item) in {3, 4} for item in data):
                data = np.array(data)

                if len(data) == 0:
                    self.points = np.empty((0, 4))
                    self.data_format = "empty_list"

                elif data.shape[1] == 3:
                    self.points = np.hstack([data, np.arange(len(data)).reshape(-1, 1)])
                    self.data_format = "list_of_lists"
                elif data.shape[1] == 4:
                    self.points = data
                    self.data_format = "list_of_lists"
            else:
                raise ValueError(
                    "Data must be a NumPy array with shape (N, 3) or (N, 4), or a list of lists with length 3 or 4.")

        else:
            self.points = None

        self.id = n_id
        self.centroid = None

        if self.points is not None:
            self.calculateCentroid()

        self.normalizePoints()

    def parseColor(self, color):
        """
        Parse the color input to support HEX or RGBA and standardize it.
        :param color: HEX string (e.g., '#DC965A') or RGBA tuple/list (e.g., [220, 150, 90, 1.0])
        :return: Dictionary with 'hex' and 'rgba' representations
        """
        if isinstance(color, str) and color.startswith("#") and len(color) in {7, 9}:
            color = color.lstrip("#")
            r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
            a = int(color[6:8], 16) / 255.0 if len(color) == 8 else 1.0
            rgba = [r, g, b, a]
            return {"hex": f"#{color}", "rgba": rgba}
        elif isinstance(color, (list, tuple)) and len(color) == 4:
            r, g, b, a = map(int, color[:3]) + [color[3]]
            hex_color = f"#{r:02X}{g:02X}{b:02X}{int(a * 255):02X}" if a < 1 else f"#{r:02X}{g:02X}{b:02X}"
            return {"hex": hex_color, "rgba": list(color)}
        else:
            raise ValueError("Color must be a HEX string or an RGBA list/tuple with 4 elements.")

    def calculateCentroid(self):
        self.centroid = np.mean(self.points[:, :3], axis=0)

    def getCentroid(self):
        return self.centroid

    def dataType(self):
        if self.data_format:
            return f"Data format is: {self.data_format}"
        return "No data loaded."


    def getPoints(self, normalized=True, include_ids=False):
        """
        Returns the point cloud data.

        :param normalized: Whether to return normalized points.
        :param include_ids: Whether to include the point IDs in the returned data.
        :return: Points as a NumPy array.
        """
        if normalized:
            points = self.normalized_points
        else:
            points = self.points[:, :3]

        if include_ids:
            return self.points

        return points


    def getColor(self):
        return self.color

    def getPointIds(self):
        return self.points[:, 3]

    def rotatePointCloud(self, angle_input):
        """
        Rotate the point cloud.
        :param angle_input: A vector of 3 elements [angle_x, angle_y, angle_z] in degrees or a 3x3 rotation matrix.
        """
        if isinstance(angle_input, (list, np.ndarray)) and len(angle_input) == 3:

            angles = np.radians(angle_input)

            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]), np.cos(angles[0])]
            ])

            Ry = np.array([
                [np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])]
            ])

            Rz = np.array([
                [np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]), np.cos(angles[2]), 0],
                [0, 0, 1]
            ])

            rotation_matrix = Rz @ Ry @ Rx

        elif isinstance(angle_input, np.ndarray) and angle_input.shape == (3, 3):
            rotation_matrix = angle_input
        else:
            raise ValueError("Input must be a 3-element vector or a 3x3 matrix.")

        self.points[:, :3] = np.dot(self.points[:, :3], rotation_matrix.T)

        self.normalizePoints()


    def applyTranslation(self, translation):
        """
        Translate the point cloud.
        :param translation: A vector of 3 elements [x, y, z] representing the translation.
        """
        if not isinstance(translation, (list, np.ndarray)) or len(translation) != 3:
            raise ValueError("Translation must be a 3-element vector.")

        self.points[:, :3] += np.array(translation)

        self.normalizePoints()

    def normalizePoints(self):
        """
        Normalize the point cloud by subtracting the centroid from all points.
        The result is stored in self.normalized_points.
        """
        centroid = np.mean(self.points[:, :3], axis=0)
        self.normalized_points = self.points[:, :3] - centroid
