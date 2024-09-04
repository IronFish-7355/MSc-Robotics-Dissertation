import numpy as np
import cv2
import cv2.aruco as aruco
import jkrc
import pyrealsense2 as rs
import math
import time

def deg_to_rad(degrees):
    return [d * math.pi / 180 for d in degrees]

def check_ret(ret, fun_name):
    if ret[0] != 0:
        print(f"{fun_name} failed, error code: {ret[0]}")
        print("Exiting program")
        exit()

def initialize_robot(robot_ip, start_pose, use_real_robot=True):
    robot = jkrc.RC(robot_ip)
    ret = robot.login()
    check_ret(ret, "login")

    if use_real_robot:
        ret = robot.power_on()
        check_ret(ret, "power on")
        ret = robot.enable_robot()
        check_ret(ret, "enable")

    print("Initializing robot arm to start pose")
    ret = robot.joint_move(start_pose, 0, True, 30)
    check_ret(ret, "moving to start pose")
    print("Robot is at start pose")
    time.sleep(5)

    return robot

robot_ip = "10.5.5.100"
use_real_robot = True
# start_pose = deg_to_rad([-180, 90, -90, -180, 90, 0])
start_pose = deg_to_rad([-180, 90, -90, -180, 90, 0])


# target_poses = [
#     deg_to_rad([-198, -33, -118, 69, 78, -215]),
#     deg_to_rad([-172, -33, -114, 63, 66, -214]),
#     deg_to_rad([-191, 0, -77, 31, -66, -190]),
#     deg_to_rad([-191, 29, -73, 31, -79, -190])
# ]

target_poses = [
    deg_to_rad([-180, 57, -100, -180, 64, -10]),
    deg_to_rad([-180, 57, -100, -180, 64, -10]),
    deg_to_rad([-180, 57, -100, -180, 64, -10]),
    deg_to_rad([-180, 57, -100, -180, 64, -10])
]

# Define the offsets for the robot arm position adjustments
arm_offset_x = 0  # in mm, adjust as necessary
arm_offset_y = 0  # in mm, adjust as necessary
arm_offset_z = 0  # in mm, adjust as necessary

# Initialize the robot
robot = initialize_robot(robot_ip, start_pose, use_real_robot)

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Align depth frames to color frames
align = rs.align(rs.stream.color)

# Load the ArUco dictionary and parameters
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

# Data recording parameters
arm_positions = []
camera_positions = []

# Define the number of calibration points per pose
num_data_points_per_pose = 20

# Function to record data at a specific pose
def record_data_at_pose(pose, num_data_points):
    robot.joint_move(pose, 0, True, 30)
    print("Moving to target pose")
    time.sleep(5)  # Wait for the robot to reach the target pose
    print("Robot has reached the target pose")
    time.sleep(1)  # Additional wait before starting data collection

    for _ in range(num_data_points):
        # Capture arm position
        pose = robot.get_tcp_position()
        if pose[0] == 0:
            adjusted_pose = pose[1].copy()
            adjusted_pose[0] += arm_offset_x
            adjusted_pose[1] += arm_offset_y
            adjusted_pose[2] += arm_offset_z

            arm_positions.append(adjusted_pose)
            print(f"Arm Pose: {adjusted_pose}")
        else:
            print(f"Failed to get robot arm position, error code: {pose[0]}")
            print("Exiting program")
            exit()

        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            markerCorners, markerIds, _ = detector.detectMarkers(gray)

            if markerIds is not None:
                intr = color_frame.profile.as_video_stream_profile().intrinsics
                intr_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
                intr_coeffs = np.array(intr.coeffs)

                marker_size = 45  # 4.5 cm = 45 mm
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(markerCorners, marker_size, intr_matrix, intr_coeffs)

                for j, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                    center = tuple(np.mean(markerCorners[j][0], axis=0).astype(int))
                    depth_value = depth_frame.get_distance(center[0], center[1])
                    x_cam, y_cam, z_cam = rs.rs2_deproject_pixel_to_point(intr, [center[0], center[1]], depth_value)

                    camera_pose = [x_cam * 1000, y_cam * 1000, z_cam * 1000, rvec[0][0], rvec[0][1], rvec[0][2]]
                    camera_positions.append(camera_pose)
                    print(f"Camera Pose: {camera_pose}")
                    break

            if markerIds is not None:
                break

        cv2.imshow('Camera View', color_image)
        cv2.waitKey(1)

        time.sleep(1)  # Wait 1 second before capturing the next data point

try:
    # Move to start pose and record data
    record_data_at_pose(start_pose, num_data_points_per_pose)

    # Move to each target pose and record data
    for target_pose in target_poses:
        record_data_at_pose(target_pose, num_data_points_per_pose)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    print("\nRecorded Data:")
    print(f"Number of arm positions: {len(arm_positions)}")
    print(f"Number of camera positions: {len(camera_positions)}")

    print("\nData in copy-paste format:")
    print("robot_poses_data = [")
    for pose in arm_positions:
        print(f"    {pose},")
    print("]")

    print("\ncamera_poses_data = [")
    for pose in camera_positions:
        print(f"    {pose},")
    print("]")

    print("Logging out...")
    ret = robot.logout()
    if ret[0] == 0:
        print("Logout succeeded")
    else:
        print(f"Logout failed, error code: {ret[0]}")

    print("Program finished")
