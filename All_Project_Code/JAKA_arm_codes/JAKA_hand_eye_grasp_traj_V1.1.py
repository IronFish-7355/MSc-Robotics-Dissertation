import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from JAKA_robot_controller_simplified import *
from Gripper_functions import GripperController
import threading
from queue import Queue, Empty



def main():
    robot_ip = "10.5.5.100"
    use_real_robot = True
    start_pose = deg_to_rad([0, -45, 90, 0, 100, 0])

    # Initialize the robot
    robot = initialize_robot(robot_ip, use_real_robot)
    robot_controller = RobotController(robot)

    gripper = GripperController(port='COM4')
    #gripper.initialize_gripper(20, 100, 1000)
    gripper_manual_flag = 1

    # Initialize RealSense camera
    pipeline, align = initialize_realsense()

    # Initialize variables for the low-pass filter
    alpha = 0.7
    filtered_coords = {}

    # Initialize selected_object_index and sample list
    selected_object_index = 0
    final_coords = None
    sampled_coords = []
    Arm_stop_pre = None

    try:
        while True:
            detected_objects, depth_image, color_image, intr = process_lab_image(pipeline, align, T_BASE_CAMERA, alpha,
                                                                                 filtered_coords)
            if detected_objects is None:
                continue

            # Create a copy of color_image for drawing
            display_image = color_image.copy()

            # Display the detected objects information
            info_image = create_info_image(detected_objects, selected_object_index)

            # Update display image
            update_display_image(display_image, detected_objects, selected_object_index, final_coords,
                                 robot_controller.get_current_state(), intr)

            # Show images using OpenCV
            cv2.imshow('Detected Objects', info_image)
            cv2.imshow('Processed Image', display_image)

            key = cv2.pollKey()
            if key != -1:
                print(f"Key pressed: {key}")
                if key == 113:  # 'q'
                    break
                elif key == 104:  # 'h'
                    if gripper_manual_flag == 1:
                        gripper.set_position(0)
                        gripper_manual_flag = 0
                    elif gripper_manual_flag == 0:
                        gripper.set_position(1000)
                        gripper_manual_flag = 1

                elif key == 115:  # 's'
                    if robot_controller.get_current_state() != State.MOVING:
                        robot_controller.set_state(State.SELECTING)
                        final_coords = None
                        sampled_coords = []
                        print("Selection mode activated.")
                        if detected_objects:
                            selected_object_index = (selected_object_index + 1) % len(detected_objects)
                            print(f"Selected object index: {selected_object_index}")
                        else:
                            print("No objects detected to select.")
                elif key == 99:  # 'c'
                    if robot_controller.get_current_state() == State.SELECTING and detected_objects:
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
                        print("Cannot confirm. Either not in selecting state or no objects detected.")
                elif key == 103:  # 'g'
                    if robot_controller.get_current_state() == State.CONFIRMATION:
                        if final_coords is not None and len(final_coords) >= 3:
                            object_rz = calculate_object_rz(final_coords[0], final_coords[1])
                            Arm_stop_pre = calculate_arm_stop_pre(final_coords[0], final_coords[1], final_coords[2],
                                                                  object_rz)
                            print(f"Calculated Arm_stop_pre: {Arm_stop_pre}")

                            # Start pre-grasp movement
                            if robot_controller.robot_grasp_pre(Arm_stop_pre):
                                print("Robot moved to pre-grasp position.")
                                robot_controller.set_state(State.PRE_GRASP)

                                # Start linear movement to object position
                                linear_target = (
                                    final_coords[0], final_coords[1], final_coords[2],
                                    0, 0, math.radians(Arm_stop_pre[5])  # rx, ry, rz (already in radians)
                                )
                                if robot_controller.robot_linear_move(linear_target):
                                    print("Robot moved to object position.")
                                    robot_controller.set_state(State.GRASP)

                                    # Perform grasping
                                    gripper.set_position(0)
                                    gripper_state = gripper.check_grasping_state()

                                    if gripper_state in [1, 3]:
                                        if robot_controller.robot_linear_move(Arm_stop_pre):
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

            # Collect samples for the selected object
            if robot_controller.get_current_state() == State.SELECTING and detected_objects:
                sampled_coords.append(detected_objects[selected_object_index][0])
                if len(sampled_coords) > 10:
                    sampled_coords.pop(0)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        gripper.stop()
        gripper.close_port()

if __name__ == "__main__":
    main()


