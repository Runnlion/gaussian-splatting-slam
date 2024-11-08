import open3d as o3d
import numpy as np

# Create a sample point cloud (or load your own point cloud)
pcd = o3d.geometry.PointCloud()
points = np.random.rand(100, 3)  # 100 random points in 3D space
pcd.points = o3d.utility.Vector3dVector(points)

# Define camera intrinsics as a 3x3 matrix
intrinsic = np.array([
    [500, 0, 320],  # fx, 0, cx
    [0, 500, 240],  # 0, fy, cy
    [0, 0, 1]       # 0, 0, 1
])

# Define camera extrinsics (identity matrix for no transformation)
extrinsics = np.eye(4)  # Camera at origin
extrinsics = np.eye(4)

from scipy.spatial.transform import Rotation as R
quaternion = [0,0,0,1]
rotation_matrix = R.from_quat(quaternion).as_matrix()
# rotation_matrix = T@rotation_matrix
theta = (np.pi/2 + np.deg2rad(0))  # Negative rotation angle in radians
print(theta)
theta = 0.0
# Rotation matrix around z-axis
R_z = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])

theta = np.deg2rad(-90)
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]
])
# Set the top-left 3x3 part of the extrinsics matrix to the rotation matrix
extrinsics[:3, :3] = R_x
extrinsics[:3, 3] = [0, 1, 1]  # Example translation

extrinsics = np.linalg.inv(extrinsics)
# Optionally set a translation (e.g., place the camera at (1, 0, 0))
# extrinsics[:3, 3] = [-x_data[0],y_data[0], z_data[0]]  # Example translation# Create the camera visualization geometry
camera = o3d.geometry.LineSet.create_camera_visualization(
    view_width_px=640,
    view_height_px=480,
    intrinsic=intrinsic,
    extrinsic=extrinsics,
    scale=0.1  # Scale of the camera visualization for better visibility
)

# Create a coordinate frame for reference
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

# Set up the Open3D viewer
viewer = o3d.visualization.Visualizer()
viewer.create_window()
viewer.add_geometry(pcd)
viewer.add_geometry(camera)
viewer.add_geometry(coordinate_frame)

# Render options
opt = viewer.get_render_option()
# opt.show_coordinate_frame = True
opt.background_color = np.asarray([0.8, 0.8, 0.8])

# Run the viewer
viewer.run()
viewer.destroy_window()
