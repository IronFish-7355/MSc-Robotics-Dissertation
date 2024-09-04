import jkrc
import time
import math
import pyrealsense2 as rs
import numpy as np
import cv2


# Functions for robot control
def deg_to_rad(degrees):
    return [d * math.pi / 180 for d in degrees]


def rad_to_deg(radians):
    return [r * 180 / math.pi for r in radians]


def check_ret(ret, fun_name):
    if ret[0] == 0:
        print(f"{fun_name} succeeded")
    else:
        print(f"{fun_name} failed, error code: {ret[0]}")
        print("Exiting program")
        exit()


def initialize_robot(robot_ip, start_pose, use_real_robot=True):
    print(f"Attempting to connect to robot, IP address: {robot_ip}")
    robot = jkrc.RC(robot_ip)

    print("Starting login...")
    ret = robot.login()
    check_ret(ret, "login")

    if use_real_robot:
        print("Powering on the robot...")
        ret = robot.power_on()
        check_ret(ret, "power on")

        print("Enabling the robot...")
        ret = robot.enable_robot()
        check_ret(ret, "enable")

    print("Retrieving joint angles...")
    angles = robot.get_joint_position()
    check_ret(angles, "retrieve joint angles")
    print(f"Current joint angles are: {rad_to_deg(angles[1])}")

    print("Retrieving end effector position...")
    pose = robot.get_tcp_position()
    check_ret(pose, "retrieve end effector position")
    print(f"Current end effector position is: {pose[1]}")

    print("Initializing robot arm to start pose")
    ret = robot.joint_move(start_pose, 0, True, 30)
    check_ret(ret, "moving to start pose")
    print("Robot is at start pose")
    time.sleep(1)

    return robot


# Constants for motion control
COORD_BASE = 0
COORD_JOINT = 1
COORD_TOOL = 2
ABS = 0
INCR = 1

# Fixing the end-effector rotation
EE_rx = deg_to_rad([0.0])
EE_ry = deg_to_rad([0.0])
EE_rz = deg_to_rad([-180.0])

#Correcting index for image coordinates to robot in mm
coods_correct_x = 10
coods_correct_y = 10
coods_correct_z = 10

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Align depth frames to color frames
align = rs.align(rs.stream.color)

# Initialize variables for the low-pass filter
alpha = 0.7
filtered_coords = {'red': None, 'yellow': None, 'lab': None}

# Transformation matrix from camera to base frame (only translation part)
T_base_camera = np.array([
    [-0.01449253, -0.89758313, 0.44060695, -555.11433787],
    [-0.99954250, 0.02470436, 0.01744938, -13.58287521],
    [-0.02654718, -0.44015249, -0.89753052, 825.45384340],
    [0.00000000, 0.00000000, 0.00000000, 1.00000000],
])


def low_pass_filter(new_value, old_value, alpha):
    if old_value is None:
        return new_value
    return alpha * old_value + (1 - alpha) * new_value


def transform_to_base(coords, T):
    # Convert coordinates from meters to millimeters
    coords_mm = coords * 1000

    # Make the coordinates homogeneous by appending 1
    coords_homogeneous = np.append(coords_mm, 1)

    # Apply the transformation matrix using the @ operator
    transformed_coords_homogeneous = T @ coords_homogeneous

    # Extract the transformed coordinates (excluding the homogeneous coordinate)
    transformed_coords = transformed_coords_homogeneous[:3]

    return transformed_coords


def process_contours(contours, color, depth_frame, intr, color_image, color_name):
    global filtered_coords
    detected_objects = []  # Reset detected objects list
    for contour in contours:
        if cv2.contourArea(contour) > 400:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate the center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2
            center = (center_x, center_y)

            # Get the depth value at the center of the bounding box
            depth_value = depth_frame.get_distance(center_x, center_y)
            if depth_value == 0:
                print(f"No depth value at center {center} for {color_name} contour.")
                continue
            x_cam, y_cam, z_cam = rs.rs2_deproject_pixel_to_point(intr, [center_x, center_y], depth_value)

            # Apply low-pass filter to the coordinates
            new_coords = np.array([x_cam, y_cam, z_cam])
            filtered_coords[color_name] = low_pass_filter(new_coords, filtered_coords[color_name], alpha)

            # Transform coordinates to base frame
            if filtered_coords[color_name] is not None:
                coords_base = transform_to_base(filtered_coords[color_name], T_base_camera)
                detected_objects.append((coords_base, (x, y, w, h), center))

                # Draw the center point
                cv2.circle(color_image, center, 5, (0, 255, 0), -1)  # Green dot for detected objects

    return detected_objects


def main():
    robot_ip = "10.5.5.100"
    use_real_robot = True
    start_pose = deg_to_rad([180, 70, -90, 180, 90, 0])

    # Initialize the robot
    robot = initialize_robot(robot_ip, start_pose, use_real_robot)

    # Initialize selected_object_index
    selected_object_index = 0

    try:
        while True:
            # Use the predefined LAB thresholds for red color
            lower_lab = np.array([0, 149, 127])
            upper_lab = np.array([255, 255, 255])

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Convert the color image to LAB
            lab_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)

            # Create masks for LAB
            mask_lab = cv2.inRange(lab_image, lower_lab, upper_lab)

            # Find contours for LAB regions
            contours_lab, _ = cv2.findContours(mask_lab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Get the camera intrinsics
            intr = color_frame.profile.as_video_stream_profile().intrinsics

            # Process LAB contours
            detected_objects = process_contours(contours_lab, (0, 255, 0), depth_frame, intr, color_image, 'lab')

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # Display the detected objects information
            info_image = np.zeros((300, 640, 3), dtype=np.uint8)
            cv2.putText(info_image, f"Objects Detected: {len(detected_objects)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            for i, (coords, bbox, center) in enumerate(detected_objects):
                x_base, y_base, z_base = coords
                x, y, w, h = bbox
                cv2.putText(info_image, f"Object {i + 1}: x: {x_base:.1f}mm, y: {y_base:.1f}mm, z: {z_base:.1f}mm",
                            (10, 60 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                if i == selected_object_index:
                    cv2.putText(info_image, f"Selected", (550, 60 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                                1)
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(color_image, f"{i + 1}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Detected Objects', info_image)

            # Show images using OpenCV
            cv2.imshow('Processed Image', color_image)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('s'):
                selected_object_index = (selected_object_index + 1) % len(detected_objects)
                print(f"Selected object index: {selected_object_index + 1}")

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()