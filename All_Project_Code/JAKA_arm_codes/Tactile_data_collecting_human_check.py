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
import csv

# Global variables
robot_controller = None
key_pressed = None
csv_writer = None
csv_file = None

Arm_pre_stop_in_degree = [-1.038, 52.640, 58.227, -90.370, 89.030, 110.870]
Arm_pre_stop = [math.radians(angle) for angle in Arm_pre_stop_in_degree]
Arm_pre_stop_in_space = None

Object_Accu_pose_in_degree = [-8.039, 53.493, 56.648, -92.784, 82.456, 110.325]
Object_Accu_pose_space = [-364.378, 357.315, 238.233, 0, 0, -3.141589771113383]

def get_folder_and_file_names(max_offset, z_offset, y_offset, ry_offset):
    folder_name = f"Z{max_offset}Y{max_offset}ry{max_offset}"
    file_name = f"Z{z_offset:+d}Y{y_offset:+d}ry{ry_offset:+.0f}"
    return folder_name, file_name

def on_key_press(key):
    global key_pressed
    try:
        key_pressed = key.char
    except AttributeError:
        key_pressed = str(key)

def create_csv_file():
    global csv_writer, csv_file
    base_dir = r"C:\Users\Hongg\Desktop\JAKA_arm_codes_backup_20240807_afternoon\DATA"
    csv_filename = f"proper_grasps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(base_dir, csv_filename)
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Z', 'Y', 'ry'])  # Write header
    print(f"CSV file created: {csv_path}")

def write_to_csv(z_offset, y_offset, ry_offset):
    global csv_writer
    if csv_writer:
        csv_writer.writerow([z_offset, y_offset, ry_offset])
        print(f"Recorded grasp position: Z={z_offset}, Y={y_offset}, ry={ry_offset}")

def main():
    global robot_controller, key_pressed, csv_writer, csv_file
    robot_ip = "10.5.5.100"
    use_real_robot = True

    print("Initializing robot and gripper...")
    # Initialize the robot
    robot = initialize_robot(robot_ip, use_real_robot)
    robot_controller = RobotController(robot)

    gripper = GripperController(port='COM3')
    gripper.initialize_gripper(20, 100, 1000)

    # Setup keyboard listener
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    # Create CSV file
    create_csv_file()

    print("\nInstructions:")
    print("- Press 'h' to start the position iteration process.")
    print("- During iteration, press 'y' to record a position as a proper grasp.")
    print("- Press 'q' at any time to quit the program.")

    try:
        robot_controller.robot.joint_move(Arm_pre_stop, 0, True, 10)
        temp_ret = robot_controller.robot.get_tcp_position()
        Arm_pre_stop_in_space = temp_ret[1]
        print("Robot initialized at:", Arm_pre_stop_in_space)

        print("\nReady to start. Press 'h' to begin iteration.")

        while True:
            if key_pressed:
                print(f"\nKey pressed: {key_pressed}")
                if key_pressed == 'q':
                    print("Quitting program...")
                    break
                elif key_pressed == 'h':
                    print("Starting position iteration process...")
                    gripper.set_position(1000)  # Keep gripper always open

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

                                    print(f"\nMoving to position: Z {i * Tac_Z_offset}, Y {k * Tac_Y_offset}, ry {math.degrees(j * Tac_ry_offset)}")

                                    # Prepare the target move with the calculated Y adjustment
                                    Object_Offset_target_increase = [0, Y_adjustment, 0, 0, 0, 0]

                                    robot_controller.robot.joint_move(ry_modify, 1, True, 80)
                                    robot_controller.robot.linear_move(Z_modify, 1, True, 80)
                                    robot_controller.robot.linear_move(Object_Offset_target_increase, 1, True, 80)

                                    print("Robot in position. Waiting for 2 seconds. Press 'y' to record as proper grasp.")
                                    # Wait for 2 seconds and check for 'y' key press
                                    start_time = time.time()
                                    while time.time() - start_time < 1.5:
                                        if key_pressed == 'y':
                                            write_to_csv(i * Tac_Z_offset, k * Tac_Y_offset, math.degrees(j * Tac_ry_offset))
                                            key_pressed = None
                                        time.sleep(0.1)

                                    # Move back to the initial position
                                    print("Returning to initial position...")
                                    Object_Offset_target_increase[1] *= -1  # Reverse Y direction
                                    robot_controller.robot.linear_move(Object_Offset_target_increase, 1, True, 80)
                                    robot_controller.robot.linear_move(Arm_pre_stop_in_space, 0, True, 80)

                    print("\nIteration process completed. Press 'h' to start again or 'q' to quit.")

                key_pressed = None  # Reset the key_pressed after processing

            time.sleep(0.01)  # Small delay to prevent busy-waiting

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        # Stop the keyboard listener
        listener.stop()

        # Cleanup
        gripper.stop()
        gripper.close_port()

        # Close CSV file
        if csv_file:
            csv_file.close()
            print("CSV file closed.")

        print("Program terminated.")

if __name__ == "__main__":
    main()