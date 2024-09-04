import serial
import time
import threading
import queue


class GripperController:
    def __init__(self, port='COM3', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.command_queue = queue.Queue()
        self.is_running = False
        self.thread = None
        self.command_lock = threading.Lock()

    def initialize_gripper(self, force, speed, position):
        """Initialize the gripper controller, open the port and set up initial settings."""
        try:

            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=2
            )
            if self.ser.is_open:
                print(f"Serial port {self.ser.port} opened successfully")

                #self.process_and_send_command("01 06 01 00 00 01")  # uncomment this to initialize the gripper via factory method (such as calibration etc)
                #self.process_and_send_command("01 06 01 00 00 A5")  # uncomment this to fully initialize the gripper via factory method, fully!
                # Set initial gripper settings here if needed
                self.set_force(force)  # Example: Set initial force
                self.set_speed(speed)  # Example: Set initial speed
                self.set_position(position)
                return True
            else:
                print(f"Failed to open serial port {self.ser.port}")
                return False
        except serial.SerialException as e:
            print(f"Serial error during initialization: {e}")
            return False

    def close_port(self):
        """Explicitly close the serial port."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed")

    def calculate_crc(self, command):
        crc = 0xFFFF
        for pos in command:
            crc ^= pos
            for _ in range(8):
                if (crc & 1) != 0:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return crc

    def append_crc(self, command):
        crc = self.calculate_crc(command)
        command.append(crc & 0xFF)
        command.append((crc >> 8) & 0xFF)
        return command

    def send_command(self, command):
        with self.command_lock:
            try:
                self.ser.write(command)
                response = self.ser.read(50)  # Adjust the number of bytes to read if necessary
                if response:
                    print(f"Received response: {response.hex()}")
                    return response
                else:
                    print("No response received")
                    return None
            except serial.SerialException as e:
                print(f"Serial error: {e}")
                return None

    def process_and_send_command(self, command, value=None):
        command_list = [int(x, 16) for x in command.split()]
        if value is not None:
            value_hex = format(value, '04X')
            value_bytes = [int(value_hex[:2], 16), int(value_hex[2:], 16)]
            command_list[-2:] = value_bytes
        command = bytearray(command_list)
        command_with_crc = self.append_crc(command)
        return self.send_command(command_with_crc)

    def set_force(self, force):
        set_force_command = "01 06 01 01 00 00"
        response = self.process_and_send_command(set_force_command, force)
        if response and len(response) >= 7:
            set_force = (response[4] << 8) | response[5]
            print(f"Force set to: {set_force}")
            return set_force
        else:
            print("Failed to set force")
            return None

    def set_position(self, position):
        set_position_command = "01 06 01 03 00 00"
        response = self.process_and_send_command(set_position_command, position)
        if response and len(response) >= 7:
            set_position = (response[4] << 8) | response[5]
            print(f"Position set to: {set_position}")
            return set_position
        else:
            print("Failed to set position")
            return None

    def set_position_fast(self, position):
        set_position_command = "01 06 01 03 00 00"
        self.process_and_send_command(set_position_command, position)



    def set_speed(self, speed):
        set_speed_command = "01 06 01 04 00 00"
        response = self.process_and_send_command(set_speed_command, speed)
        if response and len(response) >= 7:
            set_speed = (response[4] << 8) | response[5]
            print(f"Speed set to: {set_speed}")
            return set_speed
        else:
            print("Failed to set speed")
            return None

    def read_force(self):
        read_force_command = "01 03 01 01 00 01"
        response = self.process_and_send_command(read_force_command)
        if response and len(response) >= 5:
            force = (response[3] << 8) | response[4]
            print(f"Current force: {force}")
            return force
        else:
            print("Failed to read force")
            return None

    def read_position(self):
        read_position_command = "01 03 01 03 00 01"
        response = self.process_and_send_command(read_position_command)
        if response and len(response) >= 5:
            position = (response[3] << 8) | response[4]
            print(f"Current position: {position}")
            return position
        else:
            print("Failed to read position")
            return None

    def read_speed(self):
        read_speed_command = "01 03 01 04 00 01"
        response = self.process_and_send_command(read_speed_command)
        if response and len(response) >= 5:
            speed = (response[3] << 8) | response[4]
            print(f"Current speed: {speed}")
            return speed
        else:
            print("Failed to read speed")
            return None

    def gripper_control(self, force=None, position=None, speed=None):
        if force is not None:
            self.set_force(force)
        if speed is not None:
            self.set_speed(speed)
        if position is not None:
            self.set_position(position)

    def run_gripper_control(self):
        while self.is_running:
            try:
                command = self.command_queue.get(timeout=0.01)
                print(f"Processing command: {command}")
                self.gripper_control(**command)
                self.command_queue.task_done()
            except queue.Empty:
                pass

    def start(self):
        if not self.is_running and self.ser and self.ser.is_open:
            self.is_running = True
            self.thread = threading.Thread(target=self.run_gripper_control)
            self.thread.daemon = True
            self.thread.start()
            print("Gripper control thread started")
        else:
            print("Cannot start. Either already running or port not open.")

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        print("Gripper control thread stopped")

    def add_command(self, **kwargs):
        self.command_queue.put(kwargs)
        print(f"Command added to queue: {kwargs}")
        if not self.is_running:
            self.start()


    def check_grasping_state(self):
        check_state_command = "01 03 02 01 00 01"
        response = self.process_and_send_command(check_state_command)
        if response and len(response) >= 5:
            state = response[4]
            state_descriptions = {
                0: "The gripper is in motion",
                1: "The gripper has stopped moving and has not detected an object",
                2: "The gripper has stopped moving and has detected an object",
                3: "The gripper detected an object but the object has fallen"
            }
            if state in state_descriptions:
                print(f"Current state: {state_descriptions[state]}")
            else:
                print(f"Unknown state: {state}")
            return state
        else:
            print("Failed to read gripper state")
            return None


# Usage example
def main():
    gripper = GripperController(port='COM3')  # Adjust port as needed

    if gripper.initialize_gripper(force=20, speed=30, position=1000):
        try:
            gripper.start()

            # Example commands
            # gripper.add_command(force=50, speed=75, position=500)
            # time.sleep(0.5)  # Wait for the command to be processed

            gripper.set_position(0)


            gripper.set_position(1000)


            # while True:
            #     gripper.check_grasping_state()


            # gripper.set_position(1000)
            # gripper.set_position(0)
            # gripper.set_position(1000)
            # gripper.set_position(0)
            # gripper.set_position(1000)

            # Read current settings
            #gripper.read_force()
            # gripper.read_speed()
            # gripper.read_position()

            # Change position
            # gripper.add_command(position=1000)
            # time.sleep(0.5)

            # Read position again
            # gripper.read_position()


            # Wait for all commands to be processed
            gripper.command_queue.join()

        finally:
            gripper.stop()
            gripper.close_port()


if __name__ == "__main__":
    main()