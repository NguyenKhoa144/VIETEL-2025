import numpy as np
import open3d as o3d

def detect_top_parcel(pcd, depth_ud, roi=(560, 150, 300, 330)):
    """Detect top parcel in ROI, return center & normal."""
    roi_x, roi_y, roi_w, roi_h = roi
    points = np.asarray(pcd.points)

    # ROI mask on depth
    y, x = np.ogrid[:depth_ud.shape[0], :depth_ud.shape[1]]
    roi_mask = (x >= roi_x) & (x < roi_x + roi_w) & (y >= roi_y) & (y < roi_y + roi_h) & (depth_ud > 0)
    roi_depth = depth_ud[roi_mask]

    if roi_depth.size == 0:
        return None, None

    # Filter points in ROI
    valid_mask = (points[:, 0] >= roi_x / 1000) & (points[:, 1] >= roi_y / 1000) & (points[:, 0] < (roi_x + roi_w) / 1000) & (points[:, 1] < (roi_y + roi_h) / 1000)
    roi_points = points[valid_mask]

    if len(roi_points) < 50:
        return None, None

    # DBSCAN cluster
    pcd_roi = o3d.geometry.PointCloud()
    pcd_roi.points = o3d.utility.Vector3dVector(roi_points)
    labels = np.array(pcd_roi.cluster_dbscan(eps=0.02, min_points=50))

    if len(np.unique(labels)) <= 1 or -1 in labels:
        return None, None

    # Top cluster (max Z)
    cluster_means = [np.mean(roi_points[labels == i, 2]) for i in np.unique(labels) if i >= 0]
    top_id = np.argmax(cluster_means)
    top_points = roi_points[labels == top_id]

    # Center
    center = np.mean(top_points, axis=0)

    # Normal RANSAC (sửa: dùng segment_plane để fit plane)
    pcd_top = o3d.geometry.PointCloud()
    pcd_top.points = o3d.utility.Vector3dVector(top_points)
    plane_model, inliers = pcd_top.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model  # ax + by + cz + d = 0
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    # Flip inward
    camera_dir = np.array([0, 0, -1])
    if np.dot(normal, camera_dir) > 0:
        normal = -normal

    return center, normal