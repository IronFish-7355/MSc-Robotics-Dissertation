import cv2
import numpy as np
import threading
from skimage.metrics import structural_similarity as ssim

class DualCameraProcessor:
    def __init__(self, camera1_index, camera2_index):
        self.cap1 = cv2.VideoCapture(camera1_index)
        self.cap2 = cv2.VideoCapture(camera2_index)

        self.display_width = 320
        self.display_height = 240

        self.frame1 = None
        self.frame2 = None
        self.processed_frame1 = None
        self.processed_frame2 = None

        self.reference_frame1 = None
        self.reference_frame2 = None

        self.ssim_value1 = None
        self.ssim_value2 = None

        self.frame_count = 0
        self.running = False
        self.lock = threading.Lock()

    def process_frame_original(self, frame):
        return cv2.resize(frame, (self.display_width, self.display_height))

    def calculate_ssim(self):
        if self.reference_frame1 is not None and self.reference_frame2 is not None:
            if self.processed_frame1 is not None and self.processed_frame2 is not None:
                self.ssim_value1 = ssim(cv2.cvtColor(self.reference_frame1, cv2.COLOR_BGR2GRAY),
                                        cv2.cvtColor(self.processed_frame1, cv2.COLOR_BGR2GRAY))
                self.ssim_value2 = ssim(cv2.cvtColor(self.reference_frame2, cv2.COLOR_BGR2GRAY),
                                        cv2.cvtColor(self.processed_frame2, cv2.COLOR_BGR2GRAY))

    def get_frames_with_ssim(self):
        frame1 = self.processed_frame1.copy() if self.processed_frame1 is not None else None
        frame2 = self.processed_frame2.copy() if self.processed_frame2 is not None else None

        if frame1 is not None and self.ssim_value1 is not None:
            pass
            #cv2.putText(frame1, f"SSIM: {self.ssim_value1:.2f}", (10, 30),
                       #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        if frame2 is not None and self.ssim_value2 is not None:
            pass
            #cv2.putText(frame2, f"SSIM: {self.ssim_value2:.2f}", (10, 30),
                        #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame1, frame2

    def capture_and_process(self):
        while self.running:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()

            if not ret1 or not ret2:
                print("Failed to grab frame")
                break

            with self.lock:
                self.frame1 = frame1
                self.frame2 = frame2
                self.processed_frame1 = self.process_frame_original(frame1)
                self.processed_frame2 = self.process_frame_original(frame2)

                self.frame_count += 1
                if self.frame_count == 60:
                    self.reference_frame1 = self.processed_frame1.copy()
                    self.reference_frame2 = self.processed_frame2.copy()
                    print("Reference frames captured (60th frame)")

                if self.frame_count >= 3:
                    self.calculate_ssim()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.capture_and_process)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap1.release()
        self.cap2.release()

# Usage example
if __name__ == "__main__":
    processor = DualCameraProcessor(2, 1)  # Assuming camera indices 2 and 1
    processor.start()

    cv2.namedWindow('Camera 1')
    cv2.namedWindow('Camera 2')

    try:
        while True:
            frame1, frame2 = processor.get_frames_with_ssim()
            if frame1 is not None and frame2 is not None:
                cv2.imshow('Camera 1', frame1)
                cv2.imshow('Camera 2', frame2)

            # Check for window close event
            if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the Esc key
                break

    finally:
        processor.stop()
        cv2.destroyAllWindows()