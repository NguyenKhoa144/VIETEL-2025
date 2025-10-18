import cv2
import numpy as np
from pathlib import Path

def load_and_undistort(image_name: str, data_dir: Path):
    """Load RGB, depth, PLY; undistort & align."""
    import open3d as o3d  # Lazy import để tránh treo

    rgb_path = data_dir / "rgb" / image_name
    depth_path = data_dir / "depth" / image_name
    # Sửa: Relative path ./data/ply, tên file 0000.ply từ image_0000.png
    ply_filename = image_name.replace('image_', '').replace('.png', '.ply')
    ply_path = data_dir / "ply" / ply_filename

    # Load
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # m
    pcd = o3d.io.read_point_cloud(str(ply_path))

    # Color calibration
    K_color = np.array([[643.136, 0, 355.793], [0, 643.165, 253.923], [0, 0, 1]])
    dist_color = np.array([-0.056845072694397, 0.0654225662494701, -0.000869411343861069, 0.000167517995782368, -0.02095745616676633])
    h, w = rgb.shape[:2]
    new_K_color, _ = cv2.getOptimalNewCameraMatrix(K_color, dist_color, (w, h), 1, (w, h))
    rgb_ud = cv2.undistort(rgb, K_color, dist_color, None, new_K_color)

    # Depth calibration
    K_depth = np.array([[650.616, 0, 649.593], [0, 650.616, 360.942], [0, 0, 1]])
    depth_ud = cv2.undistort(depth, K_depth, np.zeros(5), None, K_depth)

    # Align PLY
    R = np.array([[0.99999, -0.00020, -0.00451], [0.00019, 0.99999, -0.00174], [0.00451, 0.00174, 0.99999]])
    t = np.array([-0.00020, -0.00451, 0.00174])
    pcd = pcd.transform(np.hstack((R, t.reshape(3,1))))

    return rgb_ud, depth_ud, pcd, new_K_color