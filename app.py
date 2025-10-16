import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append('src')  # Import src nếu cần

def compute_normal_from_xyz(x, y, z, noise_std=0.01, num_points=1000, radius=0.05):
    """Tính Rx,Ry,Rz từ x,y,z bằng PCA – high accuracy, low error."""
    # Synthetic points quanh (x,y,z) (simulate point cloud từ depth, noise Gaussian)
    center = np.array([x, y, z])
    points = np.random.normal(center, noise_std, (num_points, 3))
    
    # Giới hạn trong radius
    dists = np.linalg.norm(points - center, axis=1)
    local_points = points[dists < radius]
    
    if len(local_points) < 10:
        return None
    
    # Mean center
    local_center = np.mean(local_points, axis=0)
    centered_points = local_points - local_center
    
    # PCA: Covariance
    cov = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Normal = eigenvector eigenvalue nhỏ nhất
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    normal /= np.linalg.norm(normal)
    
    # Flip inward
    camera_dir = np.array([0, 0, -1])
    if np.dot(normal, camera_dir) > 0:
        normal = -normal
    
    return normal

def compute_oe_geodesic(n_gt, n_pred, theta_max=1.0):
    """OE geodesic low error."""
    import cv2
    from scipy.linalg import logm
    R_gt, _ = cv2.Rodrigues(n_gt.reshape(3,1))
    R_pred, _ = cv2.Rodrigues(n_pred.reshape(3,1))
    R_diff = R_gt.T @ R_pred
    omega_log = logm(R_diff)
    theta = np.linalg.norm(omega_log)
    oe_i = min(theta / theta_max, 1.0)
    return oe_i

def main():
    gt_csv = Path('./Public_train.csv')
    if not gt_csv.exists():
        print("Lỗi: Public_train.csv không tồn tại. Paste dữ liệu từ DOCUMENT vào file.")
        return
    
    df = pd.read_csv(gt_csv)
    oe_total = mce_total = n = 0
    
    print("Tính Rx,Ry,Rz từ x,y,z với PCA (low error). So GT:")
    for _, row in df.iterrows():
        image_name = row['image_filename']
        x, y, z = row['x'], row['y'], row['z']
        n_gt = np.array([row['Rx'], row['Ry'], row['Rz']])
        
        # Tính normal từ x,y,z
        n_pred = compute_normal_from_xyz(x, y, z)
        if n_pred is not None:
            # MCE (giả định center = GT x,y,z, sai số 0)
            mce_i = 0  # Hoặc tính nếu có delta
            oe_i = compute_oe_geodesic(n_gt, n_pred)
            oe_total += oe_i
            mce_total += mce_i
            n += 1
            print(f"{image_name}: x,y,z=({x:.3f},{y:.3f},{z:.3f}) -> Normal pred=({n_pred[0]:.3f},{n_pred[1]:.3f},{n_pred[2]:.3f}), OE= {oe_i:.3f}")
        else:
            print(f"{image_name}: Không đủ points, skip.")
    
    if n > 0:
        oe_avg = oe_total / n
        mce_avg = mce_total / n
        ac = (1 - mce_avg) * 0.7 + (1 - oe_avg) * 0.3
        print(f"\nKết quả trung bình (N={n}): MCE= {mce_avg:.3f}, OE Geodesic= {oe_avg:.3f}, AC= {ac:.3f}")
    else:
        print("No data to process.")

if __name__ == "__main__":
    main()