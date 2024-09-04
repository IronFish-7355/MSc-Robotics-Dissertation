from Gripper_functions import GripperController
import time


def rapid_position_change():
    gripper = GripperController(port='COM3')  # Adjust the COM port as needed
    gripper.start()

    try:
        print("Initializing gripper...")
        gripper.add_command(force=100, speed=100, position=0)
        time.sleep(1)  # Wait for initialization

        positions = [1000, 0, 1000, 0, 1000, 0, 1000, 0]

        input("Press Enter to start rapid position changes...")
        print("Starting rapid position changes. Time manually...")

        for pos in positions:
            gripper.add_command(position=pos)
            time.sleep(2)  # Minimal delay between commands

        print("Completed all position changes.")

    finally:
        print("Stopping gripper controller...")
        #gripper.stop()


if __name__ == "__main__":
    rapid_position_change()