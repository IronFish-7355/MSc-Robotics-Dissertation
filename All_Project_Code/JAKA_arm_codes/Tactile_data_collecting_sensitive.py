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
from Tac_SSIM_testing import DualCameraProcessor
import os
from datetime import datetime
import csv

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

Arm_pre_stop_in_degree = [17.252, 63.382, 62.945, -79.576, 103.824, 127.594]
Arm_pre_stop = [math.radians(angle) for angle in Arm_pre_stop_in_degree]
Arm_pre_stop_in_space = None

lift_up_pose = [0,0,50,0,0,0]

Object_Accu_pose_in_degree = [-8.039, 53.493, 56.648, -92.784, 82.456, 110.325]
Object_Accu_pose_space = [-364.378, 357.315, 238.233, 0, 0, -3.141589771113383]

SSIM_thre_1 = 0.56
SSIM_thre_2 = 0.35  # pillow

SSIM_ref_1 = 0.56
SSIM_ref_2 = 0.35

ssim_values_main_1 = 1
ssim_values_main_2 = 1

# New global variables for CSV logging
csv_filename = None
csv_file = None
csv_writer = None
grasp_index = 0

def get_folder_and_file_names(max_offset, z_offset, y_offset, ry_offset):
    # Folder name uses maximum offset values
    folder_name = f"Z{max_offset}Y{max_offset}ry{max_offset}"

    # File name includes signs and actual offset values
    file_name = f"Z{z_offset:+d}Y{y_offset:+d}ry{ry_offset:+.0f}"

    return folder_name, file_name

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

def get_csv_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"gripper_ssim_data_{timestamp}.csv"

def initialize_csv():
    global csv_filename, csv_file, csv_writer
    csv_filename = get_csv_filename()
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Grasp Index', 'Gripper Position', 'SSIM1', 'SSIM2'])

def close_csv():
    global csv_file
    if csv_file:
        csv_file.close()

def perform_grasping_task(robot_controller, gripper, dual_camera_processor):
    global grasp_index, csv_writer

    grasp_index += 1
    print(f"Performing grasping task {grasp_index}...")

    # Gradual gripper closing
    grasped = False
    grasped_position = 0
    for i_g in range(75, 0, -1):  # More gradual closing
        gripper_posi = i_g * 10
        gripper.set_position_fast(gripper_posi)

        # Update SSIM values after setting the gripper position
        ssim_values_main_1 = dual_camera_processor.ssim_value1
        ssim_values_main_2 = dual_camera_processor.ssim_value2

        print("Updated SSIM values:", ssim_values_main_1, ssim_values_main_2)

        # Log data to CSV
        csv_writer.writerow([grasp_index, gripper_posi, ssim_values_main_1, ssim_values_main_2])

        # Check if both SSIM values are below their respective thresholds
        if ssim_values_main_1 <= SSIM_thre_1 or ssim_values_main_2 <= SSIM_thre_2:
            print("Object grasped!")
            grasped = True
            grasped_position = gripper_posi
            # Add a blank row to indicate grasping
            csv_writer.writerow([grasp_index, '', '', ''])
            time.sleep(2)
            break

        time.sleep(0.1)  # Short delay between gripper movements

    if grasped:
        robot_controller.robot.linear_move(lift_up_pose, 1, True, 20)

        print("Gradually opening gripper...")
        for i_g in range(grasped_position // 10, 76):  # Start from grasped position
            gripper_posi = i_g * 10
            gripper.set_position_fast(gripper_posi)

            # Update SSIM values after setting the gripper position
            ssim_values_main_1 = dual_camera_processor.ssim_value1
            ssim_values_main_2 = dual_camera_processor.ssim_value2

            # Log data to CSV
            csv_writer.writerow([grasp_index, gripper_posi, ssim_values_main_1, ssim_values_main_2])

            time.sleep(0.1)  # Short delay between gripper movements
    else:
        print("Object not grasped. Opening gripper...")
        gripper.set_position(1000)

    time.sleep(2)

    # Move back to the initial position
    print("Returning to Arm_pre_stop_in_space...")
    robot_controller.robot.joint_move(Arm_pre_stop, 0, True, 10)
    time.sleep(1)

    print(f"Grasping task {grasp_index} completed.")

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

    # Initialize CSV file
    initialize_csv()

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
                    perform_grasping_task(robot_controller, gripper, dual_camera_processor)
                elif key_pressed == 's':
                    gripper.set_position(1000)
                    print("'s' key pressed. Add your custom event here.")
                elif key_pressed == 'c':
                    print("'c' key pressed. Add your custom event here.")
                elif key_pressed == 'g':
                    print("'g' key pressed. Add your custom event here.")

                key_pressed = None  # Reset the key_pressed after processing

            # Get the latest SSIM values
            # Print current SSIM values (for demonstration purposes)
            Thumb_SSIM = f"{ssim_values[0]:.2f}" if ssim_values[0] is not None else "N/A"
            Pillow_SSIM = f"{ssim_values[1]:.2f}" if ssim_values[1] is not None else "N/A"

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

        # Close CSV file
        close_csv()
        print(f"All grasping data saved to {csv_filename}")

if __name__ == "__main__":
    main()