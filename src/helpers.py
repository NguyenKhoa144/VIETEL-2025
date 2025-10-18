import numpy as np
import open3d as o3d

# Parameters (global for tuning)
VOXEL_SIZE = 0.008
STAT_NB = 30
STAT_STD = 1.2
RADIUS_NB = 16
RADIUS_R = 0.02
TOP_Z_BAND = 0.12
LOCAL_NORMAL_Z_THRESH = 0.7
DBSCAN_EPS = 0.04
DBSCAN_MIN_POINTS = 25
RANSAC_DIST = 0.008
RANSAC_ITERS = 2000
MIN_BOUND = np.array([-0.7312, -0.5926, -1.7230])
MAX_BOUND = np.array([1.0764, 0.5520, -0.5390])

def crop_point_cloud(pcd, min_bound, max_bound):
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    return pcd.crop(bbox)

def remove_outliers_and_downsample(pcd, voxel_size=VOXEL_SIZE):
    if not pcd.has_points():
        return pcd
    
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=STAT_NB, std_ratio=STAT_STD)
    pcd, _ = pcd.remove_radius_outlier(nb_points=RADIUS_NB, radius=RADIUS_R)
    
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    return pcd

def compute_normals(pcd, radius=0.03, max_nn=30):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    
    normals = np.asarray(pcd.normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    return pcd

def orient_normals_towards_camera(pcd, camera_location=np.array([0., 0., 0.])):
    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    view_dirs = camera_location - pts
    dot = np.einsum('ij,ij->i', view_dirs, normals)
    flip_mask = dot < 0
    normals[flip_mask] *= -1
    
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd

def select_top_horizontal_candidates(pcd_down, z_band=TOP_Z_BAND, normal_z_thresh=LOCAL_NORMAL_Z_THRESH):
    pts = np.asarray(pcd_down.points)
    if pts.shape[0] == 0:
        return None
    
    max_z = np.max(pts[:, 2])
    top_mask = pts[:, 2] > (max_z - z_band)
    
    if np.sum(top_mask) == 0:
        return None
    
    pts_top = pts[top_mask]
    normals = np.asarray(pcd_down.normals)[top_mask]
    
    good_mask = normals[:, 2] >= normal_z_thresh
    
    if np.sum(good_mask) == 0:
        good_mask = normals[:, 2] >= (normal_z_thresh - 0.2)
    
    if np.sum(good_mask) == 0:
        indices = np.where(top_mask)[0]
        return pcd_down.select_by_index(indices)
    
    top_indices = np.where(top_mask)[0]
    chosen_indices = top_indices[good_mask]
    return pcd_down.select_by_index(chosen_indices)

def cluster_and_pick_highest_z(pcd_candidate):
    pts = np.asarray(pcd_candidate.points)
    if pts.shape[0] == 0:
        return None
    
    labels = np.array(pcd_candidate.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS))
    valid_labels = [l for l in np.unique(labels) if l != -1]
    
    if not valid_labels:
        return None
    
    best_label = None
    best_meanz = -np.inf
    
    for l in valid_labels:
        idxs = np.where(labels == l)[0]
        if idxs.size == 0:
            continue
        meanz = np.mean(pts[idxs, 2])
        if meanz > best_meanz:
            best_meanz = meanz
            best_label = l
    
    if best_label is None:
        return None
    
    chosen_idxs = np.where(labels == best_label)[0]
    return pcd_candidate.select_by_index(chosen_idxs)

def lift_to_original_cloud_indices(pcd_down, pcd_orig, chosen_down_pcd):
    kd = o3d.geometry.KDTreeFlann(pcd_orig)
    orig_indices = []
    
    for pt in np.asarray(chosen_down_pcd.points):
        _, idx, _ = kd.search_knn_vector_3d(pt, 1)
        orig_indices.append(idx[0])
    
    return pcd_orig.select_by_index(orig_indices)

def fit_plane_and_get_normal(pcd_for_fit):
    pts = np.asarray(pcd_for_fit.points)
    if pts.shape[0] < 3:
        return None
    
    try:
        model, inliers = pcd_for_fit.segment_plane(
            distance_threshold=RANSAC_DIST,
            ransac_n=3,
            num_iterations=RANSAC_ITERS
        )
        normal = np.array(model[:3], dtype=float)
        
        if np.linalg.norm(normal) < 1e-8:
            return pca_normal(pts)
        
        if normal[2] < 0:
            normal = -normal
        
        return normal / np.linalg.eigh(normal)
    
    except Exception:
        return pca_normal(pts)

def pca_normal(pts):
    if pts.shape[0] < 3:
        return None
    
    m = pts.mean(axis=0)
    cov = np.cov((pts - m).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    normal = eigvecs[:, np.argmin(eigvals)]
    
    if normal[2] < 0:
        normal = -normal
    
    return normal / np.linalg.norm(normal)
