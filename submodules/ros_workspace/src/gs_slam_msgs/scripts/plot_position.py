#!/usr/bin/env python

import rospy, rosbag, csv
from gs_slam_msgs.msg import visual_merged_msg  # Replace with your message type
from geometry_msgs.msg import TransformStamped
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D scatter plots

# Initialize lists to store XYZ data
x_data, y_data, z_data = [], [], []

def extract_data_from_bag(bag_file):
    # Open the ROS bag file
    with rosbag.Bag(bag_file, 'r') as bag:
        # Iterate over all messages in the 'Visual_merged' topic
        with open('baseline_ned.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            csv_writer.writerow(['TS', 'X', 'Y', 'Z'])
            for topic, msg, t in bag.read_messages(topics=['/Visual_Merged']):
                # Extract XYZ from CameraPose
                x = msg.CameraPose.transform.translation.x
                y = msg.CameraPose.transform.translation.y
                z = msg.CameraPose.transform.translation.z

                # Store data for plotting
                x_data.append(x)
                y_data.append(y)
                z_data.append(z)

                # Optional: Print data for verification
                print(f"Time: {t} - X: {x}, Y: {y}, Z: {z}")
                csv_writer.writerow([t, x, y, z])


def plot_data():
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_data, y_data, z_data, marker='o')

    # Set labels for the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display the plot
    plt.show()

if __name__ == '__main__':
    # Provide the path to your ROS bag file
    # bag_name = 'visual_merged.bag' # 
    # bag_name = '2024-10-24-19-27-29.bag' # 
    
    bag_name = 'test1.bag' # 
    bag_file = '/home/wolftech/lxiang3.lab/Desktop/sdu6/gaussian-splatting-slam/rosbag/Latest/1106/' + bag_name  # Replace with the actual 
    
    # Extract data from the ROS bag
    extract_data_from_bag(bag_file)

    # Plot the extracted data
    plot_data()