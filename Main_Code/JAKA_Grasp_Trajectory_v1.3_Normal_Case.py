import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from JAKA_robot_controller_simplified import *
from Gripper_functions import GripperController
import threading
from queue import Queue, Empty
from pynput import keyboard
from Tactips_camera_function import DualCameraProcessor
import os
from datetime import datetime

import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from JAKA_robot_controller_simplified import *
from Gripper_functions import GripperController
import threading
from queue import Queue, Empty
from pynput import keyboard
from Tactips_camera_function import DualCameraProcessor


# Global variables
color_image = None
intr = None
latest_contours = None
contours_lock = threading.Lock()
contour_queue = Queue(maxsize=1)  # Only keep the most recent contour data
selected_object_index = 0
final_coords = None
robot_controller = None
key_pressed = None
ssim_values = (None, None)  # (ssim_value1, ssim_value2)

Z_dire_offset = 0  # mm

Arm_pre_stop_in_degree = [-1.038, 52.640, 58.227, -90.370, 89.030, 110.870]
Arm_pre_stop = [math.radians(angle) for angle in Arm_pre_stop_in_degree]
Arm_pre_stop_in_space = None

Object_Accu_pose_in_degree = [-8.039, 53.493, 56.648, -92.784, 82.456, 110.325]
Object_Accu_pose_space = [-364.378, 357.315, 238.233, 0, 0, -3.141589771113383]

Tac_Z_offset = 15  # mm
Tac_Y_offset = 15  # mm
Tac_ry_offset = math.radians(15)  # 20 degrees

SSIM_thre_1 = 0.6
SSIM_thre_2 = 0.6  # pillow

SSIM_ref_1 = 0.56
SSIM_ref_2 = 0.35

ssim_values_main_1 = 1
ssim_values_main_2 = 1


def create_dual_camera_canvas(frame1, frame2):
    # Ensure both frames are color images
    if len(frame1.shape) == 2:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
    if len(frame2.shape) == 2:
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

    # Get dimensions
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

    # Calculate canvas size
    canvas_width = w1 + w2
    canvas_height = max(h1, h2)

    # Create canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place images side by side
    canvas[:h1, :w1] = frame1
    canvas[:h2, w1:w1 + w2] = frame2

    return canvas


def on_key_press(key):
    global key_pressed
    try:
        key_pressed = key.char
    except AttributeError:
        key_pressed = str(key)


def display_thread(stop_event, dual_camera_processor):
    global color_image, latest_contours, selected_object_index, final_coords, robot_controller, intr, ssim_values

    cv2.namedWindow('Dual Camera View', cv2.WINDOW_NORMAL)

    while not stop_event.is_set():
        # Get processed frames from DualCameraProcessor
        frame1, frame2 = dual_camera_processor.get_frames_with_ssim()

        if frame1 is not None and frame2 is not None:
            canvas = create_dual_camera_canvas(frame1, frame2)
            cv2.imshow('Dual Camera View', canvas)
            cv2.waitKey(1)

        time.sleep(0.01)  # Small delay to prevent busy-waiting

    cv2.destroyAllWindows()


def record_video(dual_camera_processor, video_name, duration, fps, result_queue):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    start_time = time.time()
    frame_count = 0
    expected_frame_count = duration * fps

    while time.time() - start_time < duration:
        loop_start = time.time()

        frame1, frame2 = dual_camera_processor.get_frames_with_ssim()
        if frame1 is not None and frame2 is not None:
            canvas = create_dual_camera_canvas(frame1, frame2)
            if out is None:
                height, width = canvas.shape[:2]
                out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
            out.write(canvas)
            frame_count += 1

        # Calculate the time taken for this iteration
        processing_time = time.time() - loop_start

        # Sleep for the remaining time to maintain fps, if possible
        sleep_time = max(1 / fps - processing_time, 0)
        time.sleep(sleep_time)

    if out is not None:
        out.release()
        actual_duration = time.time() - start_time
        actual_fps = frame_count / actual_duration
        print(f"Video saved to {video_name}")
        print(f"Frames captured: {frame_count}/{expected_frame_count}")
        print(f"Actual duration: {actual_duration:.2f} seconds")
        print(f"Actual FPS: {actual_fps:.2f}")
        result_queue.put((frame_count, actual_duration, actual_fps))
    else:
        print(f"No frames were captured for {video_name}")
        result_queue.put(None)


def main():
    global selected_object_index, final_coords, robot_controller, key_pressed, color_image, intr, ssim_values
    robot_ip = "10.5.5.100"
    use_real_robot = True

    # Initialize the robot
    robot = initialize_robot(robot_ip, use_real_robot)
    robot_controller = RobotController(robot)

    gripper = GripperController(port='COM3')
    gripper.initialize_gripper(20, 100, 1000)

    # Initialize DualCameraProcessor
    dual_camera_processor = DualCameraProcessor(2, 1)  # Adjust camera indices as needed
    dual_camera_processor.start()

    # Create and start the display thread
    stop_event = threading.Event()
    display_thread_obj = threading.Thread(target=display_thread, args=(stop_event, dual_camera_processor))
    display_thread_obj.start()

    # Setup keyboard listener
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    try:
        robot_controller.robot.joint_move(Arm_pre_stop, 0, True, 10)
        temp_ret = robot_controller.robot.get_tcp_position()
        Arm_pre_stop_in_space = temp_ret[1]
        print(Arm_pre_stop_in_space)

        while True:
            if key_pressed:
                print(f"Key pressed: {key_pressed}")
                if key_pressed == 'q':
                    break
                elif key_pressed == 'h':
                    if gripper_manual_flag == 1:
                        gripper.set_position(0)
                        gripper_manual_flag = 0
                    elif gripper_manual_flag == 0:
                        gripper.set_position(1000)
                        gripper_manual_flag = 1
                elif key_pressed == 's':
                    if robot_controller.get_current_state() != State.MOVING:
                        robot_controller.set_state(State.SELECTING)
                        final_coords = None
                        sampled_coords = []
                        print("Selection mode activated.")
                        detected_objects = get_latest_contours()
                        if detected_objects:
                            selected_object_index = (selected_object_index + 1) % len(detected_objects)
                            print(f"Selected object index: {selected_object_index}")
                        else:
                            print("No objects detected to select.")
                elif key_pressed == 'c':
                    if robot_controller.get_current_state() == State.SELECTING:
                        robot_controller.set_state(State.CONFIRMATION)
                        final_coords = confirm_object(sampled_coords)
                        print(f"Final coords: {final_coords}")
                        if final_coords is not None and len(final_coords) >= 2:
                            object_rz = calculate_object_rz(final_coords[0], final_coords[1])
                            print(f"Object rz angle: {object_rz:.1f} degrees")
                            print("Confirmation mode activated. Press 'g' to start robot movement.")
                        else:
                            print("Error: Invalid final coordinates")
                            robot_controller.set_state(State.SELECTING)
                    else:
                        print("Cannot confirm. Not in selecting state.")
                elif key_pressed == 'g':
                    if robot_controller.get_current_state() == State.CONFIRMATION:
                        if final_coords is not None and len(final_coords) >= 3:
                            object_rz = calculate_object_rz(final_coords[0], final_coords[1])
                            Arm_stop_pre = calculate_arm_stop_pre(final_coords[0], final_coords[1],
                                                                  (final_coords[2] + Z_dire_offset),
                                                                  object_rz)
                            print(f"Calculated Arm_stop_pre: {Arm_stop_pre}")

                            # Start pre-grasp movement
                            if robot_controller.robot_grasp_pre(Arm_stop_pre):
                                print("Robot moved to pre-grasp position.")
                                robot_controller.set_state(State.PRE_GRASP)
                                gripper.set_position(1000)

                                # Start linear movement to object position
                                linear_target = (
                                    final_coords[0], final_coords[1], final_coords[2] + Z_dire_offset,
                                    0, 0, math.radians(Arm_stop_pre[5])  # rx, ry, rz (already in radians)
                                )
                                if robot_controller.robot_linear_move(linear_target):
                                    time.sleep(5)  # wait for robot arm finished
                                    print("Robot moved to object position.")
                                    robot_controller.set_state(State.GRASP)

                                    # Perform grasping
                                    gripper.set_position(100)
                                    gripper_state = gripper.check_grasping_state()

                                    linear_target_back = (
                                        Arm_stop_pre[0], Arm_stop_pre[1], Arm_stop_pre[2],
                                        math.radians(Arm_stop_pre[3]), math.radians(Arm_stop_pre[4]),
                                        math.radians(Arm_stop_pre[5]))  # rx, ry, rz (already in radians)

                                    if gripper_state in [1, 3]:

                                        if robot_controller.robot_linear_move(linear_target_back):
                                            time.sleep(3)
                                            print("Robot moved back to pre-grasp position.")
                                            robot_controller.set_state(State.SELECTING)
                                        else:
                                            print("Failed to move back to pre-grasp position.")
                                    elif gripper_state == 2:
                                        robot_controller.set_state(State.HARVESTING)

                                        print("Object grasped. Ready for harvesting.")
                                    else:
                                        print("Unexpected gripper state.")
                                else:
                                    print("Failed to move to object position.")
                                    robot_controller.set_state(State.SELECTING)
                            else:
                                print("Failed to move to pre-grasp position.")
                                robot_controller.set_state(State.SELECTING)
                        else:
                            print("Error: Invalid final coordinates for robot movement")
                            robot_controller.set_state(State.SELECTING)
                    else:
                        print("Cannot start movement. Not in confirmation state.")

                key_pressed = None  # Reset the key_pressed after processing

            if robot_controller.get_current_state() == State.HARVESTING:
                if robot_controller.harvest_move():
                    if robot_controller.robot_linear_move(linear_target_back):
                        robot_controller.set_state(State.BOXING)

            if robot_controller.get_current_state() == State.BOXING:
                if robot_controller.boxing_move():
                    time.sleep(3)
                    gripper.set_position(1000)

                robot_controller.set_state(State.SELECTING)

            # Collect samples for the selected object
            if robot_controller.get_current_state() == State.SELECTING:
                detected_objects = get_latest_contours_nonblocking()
                if detected_objects and selected_object_index < len(detected_objects):
                    sampled_coords.append(detected_objects[selected_object_index][0])
                    if len(sampled_coords) > 10:
                        sampled_coords.pop(0)

            time.sleep(0.01)  # Small delay to prevent busy-waiting

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop all threads
        stop_event.set()
        display_thread_obj.join()

        # Stop the keyboard listener
        listener.stop()

        # Stop the DualCameraProcessor
        dual_camera_processor.stop()

        # Cleanup
        gripper.stop()
        gripper.close_port()


if __name__ == "__main__":
    main()