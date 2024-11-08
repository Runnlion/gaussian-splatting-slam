#!/usr/bin/env python

import os
import argparse
import rosbag
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_and_save_image(msg, t, output_dir, count):
    try:
        # Get image properties
        width = msg.width
        height = msg.height
        encoding = msg.encoding
        data = msg.data

        # Map encoding to PIL mode and raw mode
        if encoding == 'rgb8':
            mode = 'RGB'
            raw_mode = 'RGB'
        elif encoding == 'rgba8':
            mode = 'RGBA'
            raw_mode = 'RGBA'
        elif encoding == 'mono8':
            mode = 'L'
            raw_mode = 'L'
        elif encoding == 'mono16':
            mode = 'I;16'
            raw_mode = 'I;16'
        elif encoding == 'bgr8':
            mode = 'RGB'
            raw_mode = 'BGR'
        elif encoding == 'bgra8':
            mode = 'RGBA'
            raw_mode = 'BGRA'
        else:
            print(f"Unsupported encoding: {encoding}")
            return

        # Create PIL Image from data
        image = Image.frombytes(mode, (width, height), data, 'raw', raw_mode, msg.step)

        # Define image file name
        img_name = f"frame_{count:06d}.png"
        img_path = os.path.join(output_dir, img_name)

        # Save image
        image.save(img_path)
        print(f"Saved image {img_name}")

    except Exception as e:
        print(f"Could not process image at time {t}: {e}")

def extract_realsense_images(bag_file, output_dir, image_topic, max_workers=4):
    """
    Extracts images from a ROS bag file and saves them to the output directory using PIL and multithreading.

    :param bag_file: Path to the ROS bag file.
    :param output_dir: Directory where images will be saved.
    :param image_topic: Topic name of the image stream in the bag file.
    :param max_workers: Maximum number of worker threads.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Opening bag file: {bag_file}")
    bag = rosbag.Bag(bag_file, 'r')
    count = 0
    actual_count = 0

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        # Read messages from the specified topic
        for topic, msg, t in bag.read_messages(topics=[image_topic]):
            # Submit the image processing task to the thread pool
            if(count %5 == 0):
                future = executor.submit(process_and_save_image, msg, t, output_dir, actual_count)
                futures.append(future)
                actual_count += 1
            count += 1

        # Wait for all futures to complete
        for future in as_completed(futures):
            pass  # You can handle exceptions here if needed

    bag.close()
    print(f"Extraction complete! {actual_count} images saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag file using PIL and multithreading.")
    parser.add_argument('bag_file', help="Path to the input ROS bag file.")
    parser.add_argument('output_dir', help="Directory to save extracted images.")
    parser.add_argument('image_topic', help="ROS topic name of the image stream.")
    parser.add_argument('--max_workers', type=int, default=36, help="Maximum number of worker threads.")

    args = parser.parse_args()

    extract_realsense_images(args.bag_file, args.output_dir, args.image_topic, args.max_workers)
