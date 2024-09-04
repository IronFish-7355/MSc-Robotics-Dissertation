import jkrc
import math
import numpy as np
import cv2
import pyrealsense2 as rs
import time
import enum
from threading import Thread, active_count
from datetime import datetime

# Constants
START_POSE_DEGREES = [0, -45, 90, 0, 100, 0]
START_POSE = [math.radians(angle) for angle in START_POSE_DEGREES]

T_BASE_CAMERA = np.array([
    [-0.01449253, -0.89758313, 0.44060695, -555.11433787],
    [-0.99954250, 0.02470436, 0.01744938, -13.58287521],
    [-0.02654718, -0.44015249, -0.89753052, 825.45384340],
    [0.00000000, 0.00000000, 0.00000000, 1.00000000],
])

T_CAMERA_BASE = np.array([
    [-0.01449300, -0.99954300, -0.02654700, 0.29180200],
    [-0.89758300, 0.02470400, -0.44015200, -134.60014800],
    [0.44060700, 0.01744900, -0.89753100, 985.69426600],
    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
])

# Enums
class State(enum.Enum):
    SELECTING = 1
    CONFIRMATION = 2
    MOVING = 3
    PRE_GRASP = 4
    GRASP = 5
    Harvesting = 6
    Boxing = 7

# Helper Functions
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

def low_pass_filter(new_value, old_value, alpha):
    if old_value is None:
        return new_value
    return alpha * old_value + (1 - alpha) * new_value

def transform_to_base(coords, T):
    coords_mm = coords * 1000
    coords_homogeneous = np.append(coords_mm, 1)
    transformed_coords_homogeneous = T @ coords_homogeneous
    transformed_coords = transformed_coords_homogeneous[:3]
    return transformed_coords

def calculate_median_coords(sampled_coords):
    sampled_coords = np.array(sampled_coords)
    median_coords = np.median(sampled_coords, axis=0)
    print(f"Median Coordinates: {median_coords}")
    return median_coords

def calculate_object_rz(x, y):
    offset_by = 45
    rz_angle_radians = math.atan2(y, x)
    rz_angle_degrees = math.degrees(rz_angle_radians)
    adjusted_rz = rz_angle_degrees + 90 + offset_by
    if adjusted_rz > 180:
        adjusted_rz -= 360
    elif adjusted_rz < -180:
        adjusted_rz += 360
    return adjusted_rz

def calculate_arm_stop_pre(x_base, y_base, z_base, object_rz):
    pre_unit = 100 # in mm as the robot arm will stop 50mm in the front of boundary box center
    pre_angle = math.radians(object_rz - 90)
    pre_distance = pre_unit

    pre_x = x_base - math.cos(pre_angle) * pre_distance
    pre_y = y_base - math.sin(pre_angle) * pre_distance
    pre_z = z_base
    rx, ry, rz = 0.0, 0.0, object_rz

    Arm_stop_pre = (pre_x, pre_y, pre_z, rx, ry, rz)
    return Arm_stop_pre

# Robot Controller Class
class RobotController:
    def __init__(self, robot):
        self.robot = robot
        self.state = State.SELECTING
        self.target_angles = None
        self.target_endPose = None
        self.movement_thread = None
        self.movement_complete = False

    def set_target_pre(self, Arm_stop_pre):
        print("Retrieving joint angles...")
        ret = self.robot.get_joint_position()
        if ret[0] != 0:
            print(f"Failed to retrieve joint angles. Error code: {ret[0]}")
            return False
        current_angles = ret[1]
        print(f"Current joint angles: {[round(math.degrees(angle), 2) for angle in current_angles]}")

        print(f"Target Arm_stop_pre: {Arm_stop_pre}")
        print("Calculating inverse kinematics...")
        Arm_stop_pre = Arm_stop_pre[:3] + tuple(map(math.radians, Arm_stop_pre[3:]))
        print(f"Input IK Arm_stop_pre: {Arm_stop_pre}")
        ret = self.robot.kine_inverse(current_angles, Arm_stop_pre)
        if ret[0] != 0:
            print(f"Failed to calculate inverse kinematics. Error code: {ret[0]}")
            print("This usually means the target position is unreachable.")
            print("Check if the Arm_stop_pre coordinates are within the robot's workspace.")
            return False
        self.target_angles = ret[1]
        print(f"Calculated target angles: {[round(math.degrees(angle), 2) for angle in self.target_angles]}")
        return True

    def start_movement_joints(self):
        if self.target_angles is None:
            print("No target set. Cannot start movement.")
            return False

        self.movement_complete = False
        self.movement_thread = Thread(target=self._move_robot_joints)
        print(f"{datetime.now()}: Starting movement thread. Active threads count: {active_count()}")
        self.movement_thread.start()
        return True

    def _move_robot_joints(self):
        print("Starting robot movement...")
        ret = self.robot.joint_move(self.target_angles, 0, False, 30)
        if ret[0] != 0:
            print(f"Failed to move to pre-grasp pose. Error code: {ret}")
        else:
            print("Robot moving to target pose controlled in joint space")
        self.movement_complete = True

    def robot_grasp_pre_non_blocking(self, Arm_stop_pre):
        print("Starting robot_grasp_pre_non_blocking...")
        if self.set_target_pre(Arm_stop_pre):
            print("Target set successfully. Starting movement...")
            return self.start_movement_joints()
        print("Failed to set target. Movement not started.")
        return False

    def set_target_linear(self, targetPose):
        print(f"Target Pose for linear movement: {targetPose}")
        self.target_endPose = targetPose
        return True

    def start_movement_linear(self):
        if self.target_endPose is None:
            print("No target set for linear movement. Cannot start movement.")
            return False

        self.movement_complete = False
        self.movement_thread = Thread(target=self._move_robot_linear)
        self.movement_thread.start()
        return True

    def _move_robot_linear(self):
        print("Starting robot linear movement...")
        ret = self.robot.linear_move(self.target_endPose, 0, True, 20) # True here is blocking the code to continue
                                                                       # execute and wait for robot finish the movement and response
        if ret[0] != 0:
            print(f"Failed to move to target pose linearly. Error code: {ret}")
        else:
            print("Robot moved to target pose in linear pattern and finished")
        self.movement_complete = True

    def robot_linear_move_non_blocking(self, target_pose):
        print("Starting robot_linear_move_non_blocking...")
        if self.set_target_linear(target_pose):
            if self.start_movement_linear():
                print("Linear movement started successfully.")
                return True
        print("Failed to start linear movement.")
        return False

    def move_to_start_position(self, start_pose):
        print("Moving robot back to start position...")

        ret = self.robot.joint_move(start_pose, 0, False, 30)
        if ret[0] != 0:
            print(f"Failed to move to start position. Error code: {ret[0]}")
            return False
        else:
            print("Robot successfully moved to start position")
            return True

    def set_state(self, new_state):
        if self.state != State.SELECTING and new_state == State.SELECTING:
            if self.move_to_start_position(START_POSE):
                self.state = new_state
                print(f"Robot state changed to: {self.state}")
            else:
                print("Failed to change state due to error in moving to start position")
        else:
            self.state = new_state
            print(f"Robot state changed to: {self.state}")

    def is_movement_complete(self):
        return self.movement_complete

    def get_current_state(self):
        return self.state

    def reset(self):
        self.target_angles = None
        self.movement_complete = False
        if self.movement_thread and self.movement_thread.is_alive():
            print("Warning: Resetting while movement is in progress.")
        self.movement_thread = None
        self.set_state(State.SELECTING)

# Image Processing Functions
def process_contours(contours, color, depth_frame, intr, color_image, color_name, filtered_coords, T_base_camera, alpha):
    detected_objects = []
    for contour in contours:
        if cv2.contourArea(contour) > 250:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            center = (center_x, center_y)
            depth_value = depth_frame.get_distance(center_x, center_y)
            if depth_value == 0:
                print(f"No depth value at center {center} for {color_name} contour.")
                continue
            x_cam, y_cam, z_cam = rs.rs2_deproject_pixel_to_point(intr, [center_x, center_y], depth_value)
            new_coords = np.array([x_cam, y_cam, z_cam])

            unique_key = f"{color_name}_{len(detected_objects)}"
            filtered_coords[unique_key] = low_pass_filter(new_coords, filtered_coords.get(unique_key, None), alpha)

            if filtered_coords[unique_key] is not None:
                coords_base = transform_to_base(filtered_coords[unique_key], T_base_camera)
                detected_objects.append((coords_base, (x, y, w, h), center))
                cv2.circle(color_image, center, 5, (0, 255, 0), -1)
    return detected_objects

def process_lab_image(pipeline, align, T_base_camera, alpha, filtered_coords):
    lower_lab = np.array([0, 149, 127])
    upper_lab = np.array([255, 255, 255])

    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None, None, None

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    lab_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
    mask_lab = cv2.inRange(lab_image, lower_lab, upper_lab)
    contours_lab, _ = cv2.findContours(mask_lab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    intr = color_frame.profile.as_video_stream_profile().intrinsics
    detected_objects = process_contours(contours_lab, (0, 255, 0), depth_frame, intr, color_image, 'lab', filtered_coords, T_base_camera, alpha)

    return detected_objects, depth_image, color_image, intr

# Display Functions
def update_display_image(display_image, detected_objects, selected_object_index, final_coords, state, intr):
    for i, (coords, bbox, center) in enumerate(detected_objects):
        x, y, w, h = bbox
        if i == selected_object_index:
            if state == State.CONFIRMATION or state == State.MOVING:
                display_final_coords(display_image, bbox, final_coords,
                                     calculate_object_rz(final_coords[0], final_coords[1]))
                Arm_stop_pre = calculate_arm_stop_pre(final_coords[0], final_coords[1], final_coords[2],
                                                      calculate_object_rz(final_coords[0], final_coords[1]))
                display_image = convert_and_display_arm_stop_pre(display_image, intr, Arm_stop_pre, T_CAMERA_BASE)
            else:
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_image, f"{i + 1}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display current state
    cv2.putText(display_image, f"State: {state.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def display_final_coords(color_image, bbox, coords, object_rz):
    x, y, w, h = bbox
    x_base, y_base, z_base = coords
    final_coords_texts = [
        f"X: {x_base:.1f}mm",
        f"Y: {y_base:.1f}mm",
        f"Z: {z_base:.1f}mm",
        f"RZ: {object_rz:.1f}deg"
    ]
    for i, text in enumerate(final_coords_texts):
        cv2.putText(color_image, text, (x, y - 100 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

def convert_and_display_arm_stop_pre(color_image, intr, Arm_stop_pre, T_camera_base):
    pre_x, pre_y, pre_z, rx, ry, rz = Arm_stop_pre
    Arm_stop_pre_homogeneous = np.array([pre_x, pre_y, pre_z, 1.0])
    t_cam = T_camera_base @ Arm_stop_pre_homogeneous
    x_2d, y_2d = rs.rs2_project_point_to_pixel(intr, t_cam[:3])
    x_2d, y_2d = int(x_2d), int(y_2d)

    if 0 <= x_2d < color_image.shape[1] and 0 <= y_2d < color_image.shape[0]:
        cv2.circle(color_image, (x_2d, y_2d), 5, (0, 255, 255), -1)
        cv2.putText(color_image, f"RZ: {rz:.1f}deg", (x_2d + 15, y_2d + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        print("Coordinates are out of bounds for drawing.")

    return color_image


def create_info_image(detected_objects, selected_object_index):
    info_image = np.zeros((300, 640, 3), dtype=np.uint8)
    cv2.putText(info_image, f"Objects Detected: {len(detected_objects)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)
    for i, (coords, bbox, center) in enumerate(detected_objects):
        x_base, y_base, z_base = coords
        cv2.putText(info_image, f"Object {i + 1}: x: {x_base:.1f}mm, y: {y_base:.1f}mm, z: {z_base:.1f}mm",
                    (10, 60 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if i == selected_object_index:
            cv2.putText(info_image, f"Selected", (550, 60 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return info_image


# Initialization Functions
def initialize_robot(robot_ip, use_real_robot=True):
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
    ret = robot.joint_move(START_POSE, 0, True, 30)
    check_ret(ret, "moving to start pose")
    print("Robot is at start pose")
    time.sleep(1)

    return robot


def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align


# Main Logic Functions
def confirm_object(sampled_coords):
    if not sampled_coords:
        print("No sampled coordinates available.")
        return None
    final_coords = calculate_median_coords(sampled_coords)
    if final_coords is None or len(final_coords) < 3:
        print("Error: Invalid final coordinates calculated.")
        return None
    print(f"Confirmed coordinates: {final_coords}")
    return final_coords

