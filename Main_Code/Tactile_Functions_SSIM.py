import cv2
import numpy as np
import threading
from skimage.metrics import structural_similarity as ssim


class DualCameraProcessor:
    def __init__(self, camera1_index, camera2_index):
        self.cap1 = cv2.VideoCapture(camera1_index)
        self.cap2 = cv2.VideoCapture(camera2_index)

        self.white_threshold1 = 196
        self.blue_threshold1 = 173
        self.kernel_size1 = 3

        self.white_threshold2 = 203
        self.blue_threshold2 = 139
        self.kernel_size2 = 3

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

    def isolate_white_and_blue(self, image, white_threshold, blue_threshold):
        b, g, r = cv2.split(image)
        mask_white = (r > white_threshold) & (g > white_threshold) & (b > white_threshold)
        mask_blue = (b > blue_threshold) & (b > r) & (b > g)
        mask = mask_white | mask_blue
        return mask.astype(np.uint8) * 255

    def apply_morphology(self, image, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return image

    def process_frame(self, frame, white_threshold, blue_threshold, kernel_size):
        binary_mask = self.isolate_white_and_blue(frame, white_threshold, blue_threshold)
        cleaned_mask = self.apply_morphology(binary_mask, kernel_size)
        return cv2.resize(cleaned_mask, (self.display_width, self.display_height))

    def calculate_ssim(self):
        if self.reference_frame1 is not None and self.reference_frame2 is not None:
            if self.processed_frame1 is not None and self.processed_frame2 is not None:
                self.ssim_value1 = ssim(self.reference_frame1, self.processed_frame1)
                self.ssim_value2 = ssim(self.reference_frame2, self.processed_frame2)

    def get_frames_with_ssim(self):
        frame1 = self.processed_frame1.copy() if self.processed_frame1 is not None else None
        frame2 = self.processed_frame2.copy() if self.processed_frame2 is not None else None

        if frame1 is not None and self.ssim_value1 is not None:
            #pass
            cv2.putText(frame1, f"SSIM: {self.ssim_value1:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        if frame2 is not None and self.ssim_value2 is not None:
            #pass
            cv2.putText(frame2, f"SSIM: {self.ssim_value2:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
                self.processed_frame1 = self.process_frame(frame1, self.white_threshold1, self.blue_threshold1,
                                                           self.kernel_size1)
                self.processed_frame2 = self.process_frame(frame2, self.white_threshold2, self.blue_threshold2,
                                                           self.kernel_size2)

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

    def set_parameters(self, camera, white_threshold, blue_threshold, kernel_size):
        with self.lock:
            if camera == 1:
                self.white_threshold1 = white_threshold
                self.blue_threshold1 = blue_threshold
                self.kernel_size1 = max(1, kernel_size)
            elif camera == 2:
                self.white_threshold2 = white_threshold
                self.blue_threshold2 = blue_threshold
                self.kernel_size2 = max(1, kernel_size)


# Usage example
if __name__ == "__main__":
    processor = DualCameraProcessor(2, 3)  # Assuming camera indices 2 and 1
    processor.start()

    cv2.namedWindow('Camera 1')
    cv2.namedWindow('Camera 2')

    cv2.createTrackbar('White', 'Camera 1', processor.white_threshold1, 255,
                       lambda x: processor.set_parameters(1, x, processor.blue_threshold1, processor.kernel_size1))
    cv2.createTrackbar('Blue', 'Camera 1', processor.blue_threshold1, 255,
                       lambda x: processor.set_parameters(1, processor.white_threshold1, x, processor.kernel_size1))
    cv2.createTrackbar('Kernel', 'Camera 1', processor.kernel_size1, 15,
                       lambda x: processor.set_parameters(1, processor.white_threshold1, processor.blue_threshold1, x))

    cv2.createTrackbar('White', 'Camera 2', processor.white_threshold2, 255,
                       lambda x: processor.set_parameters(2, x, processor.blue_threshold2, processor.kernel_size2))
    cv2.createTrackbar('Blue', 'Camera 2', processor.blue_threshold2, 255,
                       lambda x: processor.set_parameters(2, processor.white_threshold2, x, processor.kernel_size2))
    cv2.createTrackbar('Kernel', 'Camera 2', processor.kernel_size2, 15,
                       lambda x: processor.set_parameters(2, processor.white_threshold2, processor.blue_threshold2, x))

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