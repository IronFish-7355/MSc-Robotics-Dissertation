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
from Tac_SSIM_testing_no_masks import DualCameraProcessor
import os
from datetime import datetime

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

SSIM_thre_1 = 0.6
SSIM_thre_2 = 0.6  # pillow

SSIM_ref_1 = 0.56
SSIM_ref_2 = 0.35

ssim_values_main_1 = 1
ssim_values_main_2 = 1


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
                    gripper.set_position(1000)

                    for offset in [15, 10, 5]:  # Iterate through offsets
                        Tac_Z_offset = offset  # mm
                        Tac_Y_offset = offset  # mm
                        Tac_ry_offset = math.radians(offset)  # Convert to radians

                        for i in [-1, 0, 1]:
                            Z_modify = [0, 0, i * Tac_Z_offset, 0, 0, 0]

                            for j in [-1, 0, 1]:
                                ry_modify = [0, 0, 0, 0, 0, j * Tac_ry_offset]

                                # Calculate initial Y modification
                                Y_modify = Object_Accu_pose_space[1] - Arm_pre_stop_in_space[1]

                                for k in [-1, 0, 1]:
                                    Y_adjustment = Y_modify + (k * Tac_Y_offset)

                                    # Get folder and file names
                                    folder_name, file_name = get_folder_and_file_names(
                                        offset,
                                        i * Tac_Z_offset,
                                        k * Tac_Y_offset,
                                        math.degrees(j * Tac_ry_offset)
                                    )

                                    # Create folder
                                    base_dir = r"C:\Users\Hongg\Desktop\JAKA_arm_codes_backup_20240807_afternoon\DATA"
                                    full_folder_path = os.path.join(base_dir, folder_name)
                                    os.makedirs(full_folder_path, exist_ok=True)

                                    # Create full file path
                                    video_path = os.path.join(full_folder_path, f"{file_name}.mp4")

                                    print(
                                        f"Approaching target with Z {i * Tac_Z_offset}, Y {k * Tac_Y_offset}, ry {math.degrees(j * Tac_ry_offset)}")

                                    # Prepare the target move with the calculated Y adjustment
                                    Object_Offset_target_increase = [0, Y_adjustment, 0, 0, 0, 0]

                                    robot_controller.robot.joint_move(ry_modify, 1, True, 30)
                                    robot_controller.robot.linear_move(Z_modify, 1, True, 30)

                                    # Create a queue for the recording results
                                    result_queue = Queue()

                                    # Start recording in a separate thread
                                    recording_thread = threading.Thread(
                                        target=record_video,
                                        args=(dual_camera_processor, video_path, 20, 60, result_queue)
                                    )

                                    # Perform robot movements and gripper operations
                                    robot_controller.robot.linear_move(Object_Offset_target_increase, 1, False, 10)

                                    time.sleep(4)
                                    recording_thread.start()
                                    time.sleep(2)

                                    print("Performing task...")

                                    for i_g in range(75, 0, -15):
                                        gripper_posi = i_g * 10
                                        gripper.set_position_fast(gripper_posi)

                                        # Update SSIM values after setting the gripper position
                                        ssim_values_main_1 = dual_camera_processor.ssim_value1
                                        ssim_values_main_2 = dual_camera_processor.ssim_value2

                                        print("Updated SSIM values:", ssim_values_main_1, ssim_values_main_2)

                                        # Check if both SSIM values are below their respective thresholds
                                        if ssim_values_main_1 <= SSIM_thre_1 or ssim_values_main_2 <= SSIM_thre_2:
                                            time.sleep(10)
                                            break

                                        # Add a check for the bottom limit
                                        if i_g <= 20:
                                            print("Reached bottom limit. Pausing for 10 seconds.")
                                            time.sleep(10)

                                    # Handle loop completion
                                    else:
                                        print("Loop completed without breaking. Pausing for 10 seconds.")
                                        time.sleep(10)

                                    gripper.set_position(1000)
                                    time.sleep(2)

                                    # Reverse the Y direction to move back
                                    Object_Offset_target_increase[1] *= -1
                                    print("Reversing direction...")
                                    robot_controller.robot.linear_move(Object_Offset_target_increase, 1, True, 10)

                                    # Move back to the initial position
                                    print("Returning to Arm_pre_stop_in_space...")
                                    robot_controller.robot.linear_move(Arm_pre_stop_in_space, 0, True, 30)
                                    time.sleep(1)

                                    # Wait for recording thread to finish
                                    recording_thread.join()

                                    # Get the results from the queue
                                    try:
                                        recording_results = result_queue.get(timeout=1)
                                        if recording_results:
                                            frame_count, actual_duration, actual_fps = recording_results
                                            print(
                                                f"Recording completed: {frame_count} frames in {actual_duration:.2f} seconds at {actual_fps:.2f} FPS")
                                        else:
                                            print("No recording results available")
                                    except Empty:
                                        print("Timed out waiting for recording results")

                    print("'h' key pressed. Task completed.")
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


if __name__ == "__main__":
    main()