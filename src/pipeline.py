import os
import re
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
from src.helpers import *  # Import all helpers

# Paths (relative)
BASE_DIR = "."
CSV_FILE_PATH = os.path.join(BASE_DIR, "Public train.csv")
PLY_DIR = os.path.join(BASE_DIR, "data", "ply")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "submission_refined.csv")

def estimate_top_orientation(ply_path):
    """Pipeline chính: PLY → Normal vector."""
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        return None
    
    pcd = crop_point_cloud(pcd, MIN_BOUND, MAX_BOUND)
    if not pcd.has_points():
        return None
    
    pcd_down = remove_outliers_and_downsample(pcd, voxel_size=VOXEL_SIZE)
    if not pcd_down.has_points():
        return None
    
    pcd_down = compute_normals(pcd_down, radius=0.04, max_nn=35)
    
    pcd_down = orient_normals_towards_camera(pcd_down, camera_location=np.array([0., 0., 0.]))
    
    candidates = select_top_horizontal_candidates(pcd_down, z_band=TOP_Z_BAND, normal_z_thresh=LOCAL_NORMAL_Z_THRESH)
    
    if candidates is None or not candidates.has_points():
        pts = np.asarray(pcd_down.points)
        idxs = np.argsort(pts[:, 2])[::-1][:min(500, pts.shape[0])]
        candidates = pcd_down.select_by_index(idxs)
    
    cluster = cluster_and_pick_highest_z(candidates)
    if cluster is None or not cluster.has_points():
        cluster = candidates
    
    cluster_orig = lift_to_original_cloud_indices(pcd_down, pcd, cluster)
    
    normal = fit_plane_and_get_normal(cluster_orig if cluster_orig.has_points() else pcd)
    
    return normal

def process_dataset(df, ply_dir):
    """Process toàn bộ dataset."""
    results = []
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        image_filename = row['image_filename']
        x, y, z = row['x'], row['y'], row['z']
        
        result = {
            'image_filename': image_filename,
            'x': x, 'y': y, 'z': z,
            'Rx': 0.0, 'Ry': 0.0, 'Rz': 0.0
        }
        
        match = re.search(r'\d+', str(image_filename))
        if not match:
            results.append(result)
            continue
        
        ply_name = match.group(0) + ".ply"
        ply_path = os.path.join(ply_dir, ply_name)
        
        if not os.path.exists(ply_path):
            results.append(result)
            continue
        
        try:
            normal = estimate_top_orientation(ply_path)
            
            if normal is not None:
                result['Rx'] = float(normal[0])
                result['Ry'] = float(normal[1])
                result['Rz'] = float(normal[2])
        except Exception as e:
            print(f"Error processing {ply_name}: {e}")
        
        results.append(result)
    
    return pd.DataFrame(results)
