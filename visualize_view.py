import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

def parse_images_txt(file_path):
    cameras = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('#') or len(line.strip()) == 0:
                continue  # Skip comments and empty lines
            data = line.strip().split()
            # Parse image ID, quaternion, translation, and image name
            image_id = int(data[0])
            qw, qx, qy, qz = map(float, data[1:5])
            tx, ty, tz = map(float, data[5:8])
            image_name = data[9] if len(data) > 9 else None
            cameras.append({
                'id': image_id,
                'quaternion': [qx, qy, qz, qw],
                'translation': [tx, ty, tz],
                'image_name': image_name
            })
    return cameras

def visualize_cameras(cameras):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for cam in cameras:
        # Convert quaternion to rotation matrix
        quaternion = cam['quaternion']
        translation = cam['translation']
        
        rotation = R.from_quat(quaternion).as_matrix()
        
        # Define camera view direction (negative Z-axis in camera space)
        camera_dir = rotation @ np.array([0, 0, -1])  # Camera's "forward" direction
        camera_dir /= np.linalg.norm(camera_dir)  # Normalize
        
        # Plot camera position
        position = np.array(translation)
        ax.scatter(*position, color='blue', s=20)
        
        # Plot camera orientation as an arrow
        arrow_scale = 0.2  # Adjust the scale as needed
        ax.quiver(position[0], position[1], position[2],
                  camera_dir[0], camera_dir[1], camera_dir[2],
                  length=arrow_scale, color='red', normalize=True)
        
        # Optionally, label each camera by its ID or image name
        if cam['image_name']:
            ax.text(position[0], position[1], position[2], cam['image_name'], color='black')
    
    # Set axis labels and plot properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Camera Views from images.txt")
    plt.show()

# Load cameras from images.txt
file_path = '/home/wolftech/lxiang3.lab/Desktop/sdu6/gaussian-splatting-slam/rosbag/Latest/test3/sparse/0/images.txt'
cameras = parse_images_txt(file_path)

# Visualize cameras
visualize_cameras(cameras)
