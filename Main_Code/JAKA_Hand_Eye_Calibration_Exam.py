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
    ret = robot.joint_move(start_pose, 0, True, 10)
    check_ret(ret, "moving to start pose")
    print("Robot is at start pose")
    time.sleep(1)

    return robot

robot_ip = "10.5.5.100"
use_real_robot = True

start_pose = deg_to_rad([0, -45, 90, 0, 100, 0])

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

# Transformation matrix from robot base to camera
T_base_camera = np.array([
    [ -0.01449253,  -0.89758313,   0.44060695, -555.11433787],
    [ -0.99954250,   0.02470436,   0.01744938, -13.58287521],
    [ -0.02654718,  -0.44015249,  -0.89753052, 825.45384340],
    [  0.00000000,   0.00000000,   0.00000000,   1.00000000],
])

T_camera_base = np.array([
    [-0.01449300, -0.99954300, -0.02654700, 0.29180200],
    [-0.89758300,  0.02470400, -0.44015200, -134.60014800],
    [ 0.44060700,  0.01744900, -0.89753100, 985.69426600],
    [ 0.00000000,  0.00000000,  0.00000000, 1.00000000]
])


try:
    while True:
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

        # Convert the color image to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect the markers in the grayscale image
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)

        # Initialize a string to store the 6D pose information
        pose_info = ""

        # If markers are detected
        if markerIds is not None:
            # Draw the detected markers on the color image
            aruco.drawDetectedMarkers(color_image, markerCorners, markerIds)

            # Get the camera intrinsics
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            intr_matrix = np.array([
                [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
            ])
            intr_coeffs = np.array(intr.coeffs)

            # Estimate pose of each marker
            marker_size = 45  # Set the size of the markers (in millimeters)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(markerCorners, marker_size, intr_matrix, intr_coeffs)

            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                # Apply rotational offset
                original_rotation_matrix, _ = cv2.Rodrigues(rvec)
                adjusted_rotation_matrix = original_rotation_matrix
                adjusted_rvec, _ = cv2.Rodrigues(adjusted_rotation_matrix)

                # Draw axis for the marker using cv2.drawFrameAxes
                cv2.drawFrameAxes(color_image, intr_matrix, intr_coeffs, adjusted_rvec, tvec, marker_size)

                # Calculate the center of the marker
                center = np.mean(markerCorners[i][0], axis=0).astype(int)
                center = tuple(center)

                # Draw a circle at the center of the marker
                cv2.circle(color_image, center, 3, (0, 0, 255), -1)

                # Get the depth value at the center of the marker
                depth_value = depth_frame.get_distance(center[0], center[1])
                x_cam, y_cam, z_cam = rs.rs2_deproject_pixel_to_point(intr, [center[0], center[1]], depth_value)
                x_cam *= 1000  # Convert to mm
                y_cam *= 1000  # Convert to mm
                z_cam *= 1000  # Convert to mm

                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(adjusted_rvec)
                # Flatten the rotation matrix to a list for easier comparison
                rotation_matrix_flat = rotation_matrix.flatten().tolist()

                marker_id = int(markerIds[i])

                # Print the 6D pose information
                # print(f"Marker ID: {marker_id}")
                # print(f"Position: (x: {x_cam:.3f}mm, y: {y_cam:.3f}mm, z: {z_cam:.3f}mm)")
                # print(f"Rotation Vector: {adjusted_rvec.flatten()}")
                # print(f"Translation Vector: {tvec.flatten()}")
                # print(f"Rotation Matrix: {rotation_matrix}")

                # Add the 6D pose information to the string
                pose_info += f"Marker ID: {marker_id}\n"
                pose_info += f"x: {x_cam:.1f}mm, y: {y_cam:.1f}mm, z: {z_cam:.1f}mm\n"
                pose_info += f"rx: {adjusted_rvec[0][0]:.3f}, ry: {adjusted_rvec[1][0]:.3f}, rz: {adjusted_rvec[2][0]:.3f}\n"

                # Display 3D coordinates on the image
                # cv2.putText(color_image, f"x: {x_cam:.1f}mm", (center[0] + 10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(color_image, f"y: {y_cam:.1f}mm", (center[0] + 10, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(color_image, f"z: {z_cam:.1f}mm", (center[0] + 10, center[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(color_image, f"rx: {adjusted_rvec[0][0]:.3f}", (center[0] + 10, center[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(color_image, f"ry: {adjusted_rvec[1][0]:.3f}", (center[0] + 10, center[1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(color_image, f"rz: {adjusted_rvec[2][0]:.3f}", (center[0] + 10, center[1] + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Transform the marker pose from the camera frame to the robot frame
                t_cam = np.array([x_cam, y_cam, z_cam, 1])  # Already in mm, make it homogeneous

                t_base = T_base_camera @ t_cam
                #print(t_cam, t_base)


                # Extract the position in the robot frame
                x_base, y_base, z_base = t_base[:3]

                # Display transformed 3D coordinates on the image
                cv2.putText(color_image, f"x_robot: {x_base:.1f}mm", (center[0] + 10, center[1] + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(color_image, f"y_robot: {y_base:.1f}mm", (center[0] + 10, center[1] + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(color_image, f"z_robot: {z_base:.1f}mm", (center[0] + 10, center[1] + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Transform rotation matrix from camera frame to robot frame
                rotation_matrix_base = T_base_camera[:3, :3] @ rotation_matrix

                # Convert the rotation matrix back to a rotation vector
                rvec_base, _ = cv2.Rodrigues(rotation_matrix_base)

                # Display transformed rotation vector on the image
                cv2.putText(color_image, f"rx_robot: {rvec_base[0][0]:.3f}", (center[0] + 10, center[1] + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(color_image, f"ry_robot: {rvec_base[1][0]:.3f}", (center[0] + 10, center[1] + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(color_image, f"rz_robot: {rvec_base[2][0]:.3f}", (center[0] + 10, center[1] + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # _, pose_values = robot.get_tcp_position()
                # pose_text = "Robot Pose: " + ", ".join([f"{elem:.3f}" for elem in pose_values])
                # cv2.putText(color_image, pose_text, (10, color_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Get the end effector position
        _, pose_values = robot.get_tcp_position()
        x, y, z = pose_values[:3]  # Extract position values

        # Create a homogeneous coordinate
        t_robot = np.array([x, y, z, 1])

        # Transform to camera frame
        t_cam = T_camera_base @ t_robot

        # Project to 2D
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        x_2d, y_2d = rs.rs2_project_point_to_pixel(intr, t_cam[:3])

        # Convert to integer coordinates
        x_2d, y_2d = int(x_2d), int(y_2d)

        # Draw the end effector position on the image
        cv2.circle(color_image, (x_2d, y_2d), 5, (0, 255, 255), -1)  # Yellow dot

        # Display end effector coordinates
        cv2.putText(color_image, f"End effector: ({x_2d}, {y_2d})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images using OpenCV
        cv2.imshow('RealSense', images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    print("Logging out...")
    ret = robot.logout()
    if ret[0] == 0:
        print("Logout succeeded")
    else:
        print(f"Logout failed, error code: {ret[0]}")

    print("Program finished")
