import numpy as np
from scipy.spatial import KDTree


class VectorizedPath:
    def __init__(self, poses):
        self.going_forward = True
        self.poses = poses
        self.n_poses = self.poses.shape[0]
        self.planar_poses = np.zeros((self.n_poses, 3))
        self.planar_poses[:, 0] = poses[:, 0]
        self.planar_poses[:, 1] = poses[:, 1]
        self.pose_kdtree = KDTree(poses[:, :2])

        self.curvatures = np.zeros(self.n_poses)
        self.look_ahead_curvatures = np.zeros(self.n_poses)
        self.distances_to_goal = np.zeros(self.n_poses)
        self.angles = np.zeros(self.n_poses)
        self.angles_spatial_window = 0.25
        self.world_to_path_tfs_array = np.ndarray((self.n_poses, 3, 3))
        self.path_to_world_tfs_array = np.ndarray((self.n_poses, 3, 3))

    def compute_curvatures(self):
        # Steps, step sizes, forward and backward
        d_poses = np.diff(self.poses, axis=0)
        dist = np.linalg.norm(d_poses[:, :2], axis=1)

        # Backward and forward step sizes
        dpb = np.insert(dist, 0, np.nan)
        dpf = np.insert(dist, dist.shape[0], np.nan)

        # Denominator for derivatives
        deriv_denom = dpb * dpf * (dpb + dpf)

        x = self.poses[:, 0]
        y = self.poses[:, 1]

        arr_mask = np.zeros(self.n_poses, dtype=bool)
        arr_mask[1:-1] = True

        # 1st derivative
        xp, yp = np.zeros((self.n_poses, 2)).T
        xp[arr_mask] = (
            dpb[arr_mask] ** 2 * x[:-2]
            + (dpf[arr_mask] ** 2 - dpb[arr_mask] ** 2) * x[1:-1]
            - dpf[arr_mask] ** 2 * x[2:]
        )
        xp[arr_mask] /= deriv_denom[arr_mask]
        yp[arr_mask] = (
            dpb[arr_mask] ** 2 * y[:-2]
            + (dpf[arr_mask] ** 2 - dpb[arr_mask] ** 2) * y[1:-1]
            - dpf[arr_mask] ** 2 * y[2:]
        )
        yp[arr_mask] /= deriv_denom[arr_mask]

        # 2nd derivative
        xpp, ypp = np.zeros((self.n_poses, 2)).T
        xpp[arr_mask] = (
            dpb[arr_mask] ** 2 * xp[:-2]
            + (dpf[arr_mask] ** 2 - dpb[arr_mask] ** 2) * xp[1:-1]
            - dpf[arr_mask] ** 2 * xp[2:]
        )
        xpp[arr_mask] /= deriv_denom[arr_mask]
        ypp[arr_mask] = (
            dpb[arr_mask] ** 2 * yp[:-2]
            + (dpf[arr_mask] ** 2 - dpb[arr_mask] ** 2) * yp[1:-1]
            - dpf[arr_mask] ** 2 * yp[2:]
        )
        ypp[arr_mask] /= deriv_denom[arr_mask]

        # print(x.shape, xp.shape, xpp.shape, arr_mask)

        self.curvatures = np.sqrt(xpp**2 + ypp**2)

        # curvatures_list = np.gradient(self.poses[:,:2])
        # self.curvatures = np.sqrt(np.square(curvatures_list[0]) + np.square(curvatures_list[1]))

    def compute_look_ahead_curvatures(self, look_ahead_distance=1.0):
        self.look_ahead_distance_counter_array = np.zeros(self.n_poses)
        for i in range(0, self.n_poses - 1):
            path_iterator = 0
            look_ahead_distance_counter = 0
            path_curvature_sum = 0
            while look_ahead_distance_counter <= look_ahead_distance:
                if i + path_iterator + 1 == self.n_poses:
                    break
                path_curvature_sum += np.abs(self.curvatures[i + path_iterator])
                look_ahead_distance_counter += np.abs(
                    self.distances_to_goal[i + path_iterator]
                    - self.distances_to_goal[i + path_iterator + 1]
                )
                path_iterator += 1
            self.look_ahead_curvatures[i] = path_curvature_sum
            self.look_ahead_distance_counter_array[i] = look_ahead_distance_counter

    def compute_distances_to_goal(self):
        distance_to_goal = 0
        for i in range(self.n_poses - 1, 0, -1):
            distance_to_goal += np.linalg.norm(self.poses[i, :2] - self.poses[i - 1, :2])
            self.distances_to_goal[i - 1] = distance_to_goal

    def compute_angles(self):
        distance_counter = 0
        for i in range(0, self.n_poses - 1):
            j = i
            while distance_counter <= self.angles_spatial_window:
                if j == self.n_poses - 1:
                    self.angles[i:] = self.angles[i - 1]
                    break
                j += 1
                distance_counter = self.distances_to_goal[i] - self.distances_to_goal[j]
            self.angles[i] = np.arctan2(
                self.poses[j, 1] - self.poses[i, 1], self.poses[j, 0] - self.poses[i, 0]
            )
            self.planar_poses[i, 2] = self.angles[i]
            distance_counter = 0

    def compute_world_to_path_frame_tfs(self):
        path_to_world_tf = np.eye(3)
        for i in range(0, self.n_poses):
            path_to_world_tf[0, 0] = np.cos(self.angles[i])
            path_to_world_tf[0, 1] = -np.sin(self.angles[i])
            path_to_world_tf[0, 2] = self.poses[i, 0]
            path_to_world_tf[1, 0] = np.sin(self.angles[i])
            path_to_world_tf[1, 1] = np.cos(self.angles[i])
            path_to_world_tf[1, 2] = self.poses[i, 1]
            self.path_to_world_tfs_array[i, :, :] = path_to_world_tf
            self.world_to_path_tfs_array[i, :, :] = np.linalg.inv(path_to_world_tf)

    def compute_metrics(self, path_look_ahead_distance):
        self.compute_distances_to_goal()
        self.compute_curvatures()
        self.compute_look_ahead_curvatures(path_look_ahead_distance)
        self.compute_angles()
        self.compute_world_to_path_frame_tfs()
        return None

    def compute_orthogonal_projection(
        self, pose, last_projection_id, query_knn, query_radius
    ):
        orthogonal_projection_dists, orthogonal_projection_ids = self.pose_kdtree.query(
            pose[:2], k=query_knn, distance_upper_bound=query_radius
        )
        # orthogonal_projection_dists, orthogonal_projection_ids = self.pose_kdtree.query_ball_point(pose[:2], query_radius, return_sorted=False)
        return orthogonal_projection_dists, orthogonal_projection_ids


# TODO: find a way to split path into multiple directional paths to switch robot direction


if __name__ == "__main__":
    test_path_poses = np.load("tests/traj_a_int.npy")
    test_path = VectorizedPath(test_path_poses)
    test_path.compute_curvatures()