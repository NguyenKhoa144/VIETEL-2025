import numpy as np
import pandas as pd
import cv2  # Import cv2 cho Rodrigues
from scipy.linalg import logm  # Import scipy cho logm (geodesic)

def compute_oe_geodesic(n_gt, n_pred, theta_max=1.0):
    """OE với Geodesic Distance (low error for small angles)."""
    # Chuyển normal sang rotation matrix R
    R_gt, _ = cv2.Rodrigues(n_gt.reshape(3,1))
    R_pred, _ = cv2.Rodrigues(n_pred.reshape(3,1))
    R_diff = R_gt.T @ R_pred
    # Geodesic angle
    omega_log = logm(R_diff)
    theta = np.linalg.norm(omega_log)
    oe_i = min(theta / theta_max, 1.0)
    return oe_i

def compute_metrics(pred_csv, gt_csv):
    """Tính MCE, OE geodesic, AC."""
    pred_df = pd.read_csv(pred_csv)
    gt_df = pd.read_csv(gt_csv).set_index('image_filename')
    mce = oe = n = 0
    for _, row in pred_df.iterrows():
        img = row['image_filename']
        if img in gt_df.index:
            gt_row = gt_df.loc[img]
            # MCE: Euclidean 3D / 0.05m
            delta = np.array([row['x'] - gt_row['x'], row['y'] - gt_row['y'], row['z'] - gt_row['z']])
            e_i = np.linalg.norm(delta) / 50.0  # 0.05m = 50mm
            mce += e_i
            # OE geodesic
            n_gt = np.array([gt_row['Rx'], gt_row['Ry'], gt_row['Rz']])
            n_pred = np.array([row['Rx'], row['Ry'], row['Rz']])
            oe += compute_oe_geodesic(n_gt, n_pred)
            n += 1
    if n > 0:
        mce /= n
        oe /= n
        ac = (1 - mce) * 0.7 + (1 - oe) * 0.3
        print(f"MCE: {mce:.3f}, OE Geodesic: {oe:.3f}, AC: {ac:.3f}")
    else:
        print("No matching images.")