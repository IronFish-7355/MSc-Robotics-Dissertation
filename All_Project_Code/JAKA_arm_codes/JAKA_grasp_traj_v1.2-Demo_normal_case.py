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

Z_dire_offset = 0  # mm


def create_large_canvas_custom(windows):
    assert len(windows) == 3, "This layout requires exactly 3 windows"

    main_image, camera1, camera2 = windows

    # Get dimensions
    main_h, main_w = main_image.shape[:2]
    cam1_h, cam1_w = camera1.shape[:2]
    cam2_h, cam2_w = camera2.shape[:2]

    # Calculate canvas size
    canvas_width = max(main_w, cam1_w + cam2_w)
    canvas_height = main_h + max(cam1_h, cam2_h)

    # Create canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place main image at the top
    canvas[:main_h, :main_w] = main_image

    # Place camera images side by side below
    y_offset = main_h
    canvas[y_offset:y_offset + cam1_h, :cam1_w] = camera1
    canvas[y_offset:y_offset + cam2_h, cam1_w:cam1_w + cam2_w] = camera2

    return canvas


def image_processing_thread(pipeline, align, stop_event):
    global latest_contours, selected_object_index, final_coords, robot_controller, color_image, intr
    alpha = 0.7
    filtered_coords = {}

    while not stop_event.is_set():
        try:
            detected_objects, depth_image, color_image, intr = process_lab_image(pipeline, align, T_BASE_CAMERA, alpha,
                                                                                 filtered_coords)

            if detected_objects is not None:
                # Update latest contours in a thread-safe manner
                with contours_lock:
                    latest_contours = detected_objects

                # Put the latest contours in the queue, replacing old data if necessary
                if not contour_queue.full():
                    contour_queue.put(detected_objects)
                else:
                    try:
                        contour_queue.get_nowait()
                        contour_queue.put(detected_objects)
                    except Empty:
                        pass

            time.sleep(0.01)  # Control frame rate
        except Exception as e:
            print(f"Error in image processing thread: {e}")


def get_latest_contours():
    global latest_contours
    with contours_lock:
        return latest_contours.copy() if latest_contours is not None else None


def get_latest_contours_nonblocking():
    try:
        return contour_queue.get_nowait()
    except Empty:
        return None


def on_key_press(key):
    global key_pressed
    try:
        key_pressed = key.char
    except AttributeError:
        key_pressed = str(key)


def display_thread(stop_event):
    global color_image, latest_contours, selected_object_index, final_coords, robot_controller, intr

    cv2.namedWindow('All Windows', cv2.WINDOW_NORMAL)
    dual_camera_processor = DualCameraProcessor(2, 1)  # Adjust camera indices as needed
    dual_camera_processor.start()

    while not stop_event.is_set():
        # Get processed frames from DualCameraProcessor
        frame1, frame2 = dual_camera_processor.get_frames_with_ssim()

        # Get the main processed image
        detected_objects = get_latest_contours()

        if detected_objects is not None and color_image is not None:
            display_image = color_image.copy()
            info_image = create_info_image(detected_objects, selected_object_index)
            if intr is not None:
                update_display_image(display_image, detected_objects, selected_object_index, final_coords,
                                     robot_controller.get_current_state(), intr)
        else:
            display_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(display_image, "No image available", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            info_image = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(info_image, "No info available", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Prepare windows
        windows = [display_image]
        if frame1 is not None:
            windows.append(frame1 if len(frame1.shape) == 3 else cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR))
        if frame2 is not None:
            windows.append(frame2 if len(frame2.shape) == 3 else cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR))

        # Ensure we have exactly 3 windows
        if len(windows) == 3:
            large_canvas = create_large_canvas_custom(windows)
            cv2.imshow('All Windows', large_canvas)
            cv2.waitKey(1)

        time.sleep(0.01)  # Small delay to prevent busy-waiting

    dual_camera_processor.stop()
    cv2.destroyAllWindows()


def main():
    global selected_object_index, final_coords, robot_controller, key_pressed, color_image, intr
    robot_ip = "10.5.5.100"
    use_real_robot = True
    start_pose = deg_to_rad([0, -45, 90, 0, 100, 0])

    # Initialize the robot
    robot = initialize_robot(robot_ip, use_real_robot)
    robot_controller = RobotController(robot)

    gripper = GripperController(port='COM3')
    gripper.initialize_gripper(20, 100, 1000)
    gripper_manual_flag = 1

    # Initialize RealSense camera
    pipeline, align = initialize_realsense()

    sampled_coords = []
    Arm_stop_pre = None

    # Create and start the image processing thread
    stop_event = threading.Event()
    image_thread = threading.Thread(target=image_processing_thread, args=(pipeline, align, stop_event))
    image_thread.start()

    # Create and start the display thread
    display_thread_obj = threading.Thread(target=display_thread, args=(stop_event,))
    display_thread_obj.start()

    # Setup keyboard listener
    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    try:
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
        image_thread.join()
        display_thread_obj.join()

        # Stop the keyboard listener
        listener.stop()

        # Cleanup
        pipeline.stop()
        gripper.stop()
        gripper.close_port()


if __name__ == "__main__":
    main()