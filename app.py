import os
import re
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

# ==================== PARAMETERS ====================
# Preprocessing
VOXEL_SIZE = 0.01
STAT_NB = 30
STAT_STD = 1.2
RADIUS_NB = 16
RADIUS_R = 0.02

# Target region selection
TARGET_RADIUS = 0.15  # Bán kính tìm kiếm quanh (x,y,z)
TARGET_RADIUS_FALLBACK = 0.25  # Bán kính mở rộng nếu tìm ít điểm
MIN_POINTS_IN_REGION = 50

# Top surface selection
TOP_Z_BAND = 0.12
LOCAL_NORMAL_Z_THRESH = 0.7

# Clustering
DBSCAN_EPS = 0.04
DBSCAN_MIN_POINTS = 25

# Plane fitting
RANSAC_DIST = 0.006
RANSAC_ITERS = 500
MIN_POINTS_FINAL = 100

# ROI boundaries (optional - có thể bỏ nếu không cần)
MIN_BOUND = np.array([-0.7312, -0.5926, -1.7230])
MAX_BOUND = np.array([1.0764, 0.5520, -0.5390])
USE_ROI_CROP = True  # Set False nếu không muốn crop

# ==================== PATHS ====================
BASE_DIR = "."
CSV_FILE_PATH = os.path.join(BASE_DIR, "Public train.csv")
PLY_DIR = os.path.join(BASE_DIR, "data", "ply")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "submission_refined.csv")


# ==================== HELPER FUNCTIONS ====================

def crop_point_cloud(pcd, min_bound, max_bound):
    """Crop point cloud theo bounding box"""
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    return pcd.crop(bbox)


def remove_outliers_and_downsample(pcd, voxel_size=VOXEL_SIZE):
    """Loại bỏ outliers và downsample"""
    if not pcd.has_points():
        return pcd
    
    # Statistical outlier removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=STAT_NB, std_ratio=STAT_STD)
    
    # Radius outlier removal
    pcd, _ = pcd.remove_radius_outlier(nb_points=RADIUS_NB, radius=RADIUS_R)
    
    # Voxel downsampling
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    return pcd


def compute_normals(pcd, radius=0.03, max_nn=30):
    """Tính normal vectors cho point cloud"""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    
    # Normalize normals
    normals = np.asarray(pcd.normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms
    pcd.normals = o3d.utility.Vector3dVector(normals)
    
    return pcd


def orient_normals_towards_camera(pcd, camera_location=np.array([0., 0., 0.])):
    """Định hướng normals về phía camera"""
    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    view_dirs = camera_location - pts
    dot = np.einsum('ij,ij->i', view_dirs, normals)
    
    # Flip normals hướng ra xa camera
    flip_mask = dot < 0
    normals[flip_mask] *= -1
    
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def select_points_near_target(pcd, target_xyz, radius=TARGET_RADIUS):
    """
    Chọn các điểm gần target_xyz trong bán kính radius
    Đây là bước QUAN TRỌNG để xác định đúng bưu kiện
    """
    points = np.asarray(pcd.points)
    
    # Tính khoảng cách đến target
    distances = np.linalg.norm(points - target_xyz, axis=1)
    
    # Lọc điểm trong bán kính
    mask = distances < radius
    
    # Nếu quá ít điểm, mở rộng bán kính
    if np.sum(mask) < MIN_POINTS_IN_REGION:
        radius = TARGET_RADIUS_FALLBACK
        mask = distances < radius
    
    if np.sum(mask) == 0:
        return None
    
    indices = np.where(mask)[0]
    return pcd.select_by_index(indices)


def select_top_horizontal_candidates(pcd, z_band=TOP_Z_BAND, normal_z_thresh=LOCAL_NORMAL_Z_THRESH):
    """
    Chọn điểm ứng viên cho mặt trên:
    - Nằm trong băng tần z_band từ điểm cao nhất
    - Có normal gần thẳng đứng
    """
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return None
    
    # Tìm max z
    max_z = np.max(pts[:, 2])
    
    # Lọc điểm trong top z_band
    top_mask = pts[:, 2] > (max_z - z_band)
    if np.sum(top_mask) == 0:
        return None
    
    pts_top = pts[top_mask]
    normals = np.asarray(pcd.normals)[top_mask]
    
    # Lọc theo normal z (gần thẳng đứng)
    good_mask = normals[:, 2] >= normal_z_thresh
    
    # Fallback: nới lỏng threshold
    if np.sum(good_mask) == 0:
        good_mask = normals[:, 2] >= (normal_z_thresh - 0.2)
    
    if np.sum(good_mask) == 0:
        # Trả về tất cả điểm top nếu không tìm được
        indices = np.where(top_mask)[0]
        return pcd.select_by_index(indices)
    
    top_indices = np.where(top_mask)[0]
    chosen_indices = top_indices[good_mask]
    
    return pcd.select_by_index(chosen_indices)


def cluster_and_pick_highest_z(pcd_candidate):
    """
    Cluster các điểm ứng viên và chọn cluster có trung bình z cao nhất
    """
    pts = np.asarray(pcd_candidate.points)
    if pts.shape[0] == 0:
        return None
    
    # DBSCAN clustering
    labels = np.array(pcd_candidate.cluster_dbscan(
        eps=DBSCAN_EPS, 
        min_points=DBSCAN_MIN_POINTS
    ))
    
    valid_labels = [l for l in np.unique(labels) if l != -1]
    
    if not valid_labels:
        return None
    
    # Tìm cluster có mean z cao nhất
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
    """
    Map điểm từ downsampled cloud về original cloud
    """
    kd = o3d.geometry.KDTreeFlann(pcd_orig)
    orig_indices = []
    
    for pt in np.asarray(chosen_down_pcd.points):
        _, idx, _ = kd.search_knn_vector_3d(pt, 1)
        orig_indices.append(idx[0])
    
    return pcd_orig.select_by_index(orig_indices)


def pca_normal(pts):
    """Tính normal bằng PCA (fallback method)"""
    if pts.shape[0] < 3:
        return None
    
    m = pts.mean(axis=0)
    cov = np.cov((pts - m).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Normal = eigenvector với eigenvalue nhỏ nhất
    normal = eigvecs[:, np.argmin(eigvals)]
    
    # Đảm bảo hướng lên (z > 0)
    if normal[2] < 0:
        normal = -normal
    
    return normal / np.linalg.norm(normal)


def fit_plane_and_get_normal(pcd_for_fit):
    """
    Fit mặt phẳng và trích xuất normal vector
    Dùng RANSAC, fallback về PCA nếu fail
    """
    pts = np.asarray(pcd_for_fit.points)
    if pts.shape[0] < 3:
        return None
    
    try:
        # RANSAC plane fitting
        model, inliers = pcd_for_fit.segment_plane(
            distance_threshold=RANSAC_DIST,
            ransac_n=3,
            num_iterations=RANSAC_ITERS
        )
        
        normal = np.array(model[:3], dtype=float)
        
        # Kiểm tra valid
        if np.linalg.norm(normal) < 1e-8:
            return pca_normal(pts)
        
        # Đảm bảo hướng lên
        if normal[2] < 0:
            normal = -normal
        
        return normal / np.linalg.norm(normal)
    
    except Exception as e:
        # Fallback về PCA
        return pca_normal(pts)


# ==================== MAIN PIPELINE ====================

def estimate_top_orientation_at_location(ply_path, target_x, target_y, target_z):
    """
    Pipeline chính: Ước lượng vector pháp tuyến tại vị trí (x,y,z)
    
    Args:
        ply_path: Đường dẫn file PLY
        target_x, target_y, target_z: Tọa độ từ thành viên 1
    
    Returns:
        normal: Vector pháp tuyến (Rx, Ry, Rz) hoặc None nếu fail
    """
    target_xyz = np.array([target_x, target_y, target_z])
    
    # 1. Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        return None
    
    # 2. (Optional) Crop ROI
    if USE_ROI_CROP:
        pcd = crop_point_cloud(pcd, MIN_BOUND, MAX_BOUND)
        if not pcd.has_points():
            return None
    
    # 3. Denoise và downsample
    pcd_down = remove_outliers_and_downsample(pcd, voxel_size=VOXEL_SIZE)
    if not pcd_down.has_points():
        return None
    
    # 4. Compute normals
    pcd_down = compute_normals(pcd_down, radius=0.04, max_nn=35)
    pcd_down = orient_normals_towards_camera(pcd_down, camera_location=np.array([0., 0., 0.]))
    
    # ===== BƯỚC QUAN TRỌNG =====
    # 5. Chọn vùng gần target (x,y,z)
    pcd_near_target = select_points_near_target(pcd_down, target_xyz, radius=TARGET_RADIUS)
    
    if pcd_near_target is None or not pcd_near_target.has_points():
        # Fallback: Nếu không tìm thấy điểm gần target, dùng toàn bộ
        print(f"  → Warning: Không tìm thấy điểm gần target {target_xyz}, dùng toàn bộ cloud")
        pcd_near_target = pcd_down
    
    # 6. Trong vùng target, chọn mặt trên
    candidates = select_top_horizontal_candidates(
        pcd_near_target, 
        z_band=TOP_Z_BAND, 
        normal_z_thresh=LOCAL_NORMAL_Z_THRESH
    )
    
    if candidates is None or not candidates.has_points():
        # Fallback: Lấy điểm cao nhất trong vùng
        pts = np.asarray(pcd_near_target.points)
        idxs = np.argsort(pts[:, 2])[::-1][:min(500, pts.shape[0])]
        candidates = pcd_near_target.select_by_index(idxs)
    
    # 7. Cluster để loại bỏ noise
    cluster = cluster_and_pick_highest_z(candidates)
    if cluster is None or not cluster.has_points():
        cluster = candidates
    
    # 8. Lift về original resolution để có độ chính xác cao
    cluster_orig = lift_to_original_cloud_indices(pcd_down, pcd, cluster)
    
    # 9. Fit plane và trích xuất normal
    if cluster_orig.has_points():
        normal = fit_plane_and_get_normal(cluster_orig)
    else:
        normal = fit_plane_and_get_normal(pcd)
    
    return normal


# ==================== DATASET PROCESSING ====================

def process_dataset(df, ply_dir):
    """
    Process toàn bộ dataset
    
    Args:
        df: DataFrame chứa image_filename, x, y, z
        ply_dir: Thư mục chứa file PLY
    
    Returns:
        results_df: DataFrame với columns [image_filename, x, y, z, Rx, Ry, Rz]
    """
    results = []
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing parcels"):
        image_filename = row['image_filename']
        x, y, z = row['x'], row['y'], row['z']
        
        # Khởi tạo result với giá trị mặc định
        result = {
            'image_filename': image_filename,
            'x': x, 'y': y, 'z': z,
            'Rx': 0.0, 'Ry': 0.0, 'Rz': 0.0
        }
        
        # Extract PLY filename từ image filename
        match = re.search(r'\d+', str(image_filename))
        if not match:
            print(f"  → Skip: Không parse được số từ {image_filename}")
            results.append(result)
            continue
        
        ply_name = match.group(0) + ".ply"
        ply_path = os.path.join(ply_dir, ply_name)
        
        if not os.path.exists(ply_path):
            print(f"  → Skip: Không tìm thấy {ply_path}")
            results.append(result)
            continue
        
        try:
            # Estimate normal vector tại vị trí (x,y,z)
            normal = estimate_top_orientation_at_location(ply_path, x, y, z)
            
            if normal is not None:
                result['Rx'] = float(normal[0])
                result['Ry'] = float(normal[1])
                result['Rz'] = float(normal[2])
            else:
                print(f"  → Warning: Không tính được normal cho {ply_name}")
        
        except Exception as e:
            print(f"  → Error processing {ply_name}: {e}")
        
        results.append(result)
    
    return pd.DataFrame(results)


# ==================== MAIN ====================

def main():
    """Main function"""
    print("=" * 60)
    print("PARCEL ORIENTATION ESTIMATION")
    print("=" * 60)
    
    # Kiểm tra files
    if not os.path.exists(CSV_FILE_PATH):
        print(f"ERROR: Không tìm thấy {CSV_FILE_PATH}")
        return
    
    if not os.path.exists(PLY_DIR):
        print(f"ERROR: Không tìm thấy thư mục {PLY_DIR}")
        return
    
    # Load CSV
    print(f"\n1. Loading CSV: {CSV_FILE_PATH}")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"   → Loaded {len(df)} samples")
    print(f"   → Columns: {list(df.columns)}")
    
    # Process dataset
    print(f"\n2. Processing point clouds...")
    results_df = process_dataset(df, PLY_DIR)
    
    # Statistics
    non_zero_count = sum(1 for r in results_df['Rz'] if r > 0)
    success_rate = (non_zero_count / len(results_df)) * 100
    
    print(f"\n3. Results:")
    print(f"   → Total samples: {len(results_df)}")
    print(f"   → Successfully processed: {non_zero_count}")
    print(f"   → Success rate: {success_rate:.2f}%")
    
    # Save results
    print(f"\n4. Saving to {OUTPUT_CSV_PATH}")
    results_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.6f')
    
    # Preview
    print(f"\n5. Preview (first 5 rows):")
    print(results_df.head())
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()