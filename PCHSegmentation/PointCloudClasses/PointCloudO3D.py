import open3d as o3d
import numpy as np
from PCHSegmentation.PointCloudClasses.PointCloudGeneric import PointCloudGeneric


class PointCloudO3D(PointCloudGeneric):
    def __init__(self, points=None, standard_id="Bunny", n_p=10000, n_id=0, color="#DC965A"):
        """
        Initialize the PointCloudO3D object and create a point cloud list.

        :param points: List of points to initialize the point cloud.
        :param standard_id: The ID of the standard model ('Bunny', 'Armadillo', or 'Knot').
        :param n_p: Number of points to sample from the mesh.
        :param n_id: ID for the PointCloud object.
        :param color: Color in HEX or RGBA.
        """
        pc_data = [[0, 0, 0]]

        if points is not None:
            super().__init__(data=np.asarray(points), n_id=n_id, color=color)
        else:
            if standard_id == "Armadillo":
                armadillo_data = o3d.data.ArmadilloMesh()
                mesh = o3d.io.read_triangle_mesh(armadillo_data.path)
                armadillo_point_cloud = mesh.sample_points_uniformly(number_of_points=n_p)
                pc_data = armadillo_point_cloud

            elif standard_id == "Bunny":
                bunny_data = o3d.data.BunnyMesh()
                mesh = o3d.io.read_triangle_mesh(bunny_data.path)
                bunny_point_cloud = mesh.sample_points_uniformly(number_of_points=n_p)
                pc_data = bunny_point_cloud

            elif standard_id == "Knot":
                knot_data = o3d.data.KnotMesh()
                mesh = o3d.io.read_triangle_mesh(knot_data.path)
                knot_cloud = mesh.sample_points_uniformly(number_of_points=n_p)
                pc_data = knot_cloud

            super().__init__(data=np.asarray(pc_data.points), n_id=n_id, color=color)

        


    def getPointList(self):
        """
        Return the current point list.

        :return: Numpy array with shape (N, 4), where each row is [x, y, z, p_id].
        """
        return self.point_list

    def rotatePointCloud(self, angle_input):
        """
        Rotate the point cloud and update the point list.

        :param angle_input: A vector of 3 elements [angle_x, angle_y, angle_z] in degrees or a 3x3 rotation matrix.
        """
        super().rotatePointCloud(angle_input)
        

    def applyTranslation(self, translation):
        """
        Translate the point cloud and update the point list.

        :param translation: A vector of 3 elements [x, y, z] representing the translation.
        """
        super().applyTranslation(translation)
        

    def normalizePoints(self):
        """
        Normalize the point cloud by subtracting the centroid from all points.
        Update the point list to reflect the normalized points.
        """
        super().normalizePoints()
        



