

class PointCloudDatabase:
    def __init__(self, point_cloud):
        """
        Initialize the PointCloudDatabase to store segments at different levels of segmentation.
        """
        self.root = point_cloud
        self.database = [[self.root]]

    def subdivideLevel(self, computeMethod, params, n_level):

        level = self.database[n_level]

        new_level = []

        for seg in level:
            if "Slices" in params:
                new_seg = computeMethod(seg, params["Slices"])
            elif "Direction" in params:
                new_seg = computeMethod(seg, params["Direction"])
            else:
                new_seg = computeMethod(seg)

            new_level += new_seg

        self.database.append(new_level)


class AdaptivePointCloudDatabase (PointCloudDatabase):
    def __init__(self, point_cloud):
        super().__init__(point_cloud)

    def subdivideLevel(self, computeMethod, params, n_level, min_cell_density=100):

        level = self.database[n_level]

        new_level = []

        for seg in level:
            if len(seg.getPoints()) <= min_cell_density:
                new_level += [seg]
                continue

            if "Slices" in params:
                new_seg = computeMethod(seg, params["Slices"])
            elif "Direction" in params:
                new_seg = computeMethod(seg, params["Direction"])
            else:
                new_seg = computeMethod(seg)

            new_level += new_seg

        self.database.append(new_level)
