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

        self.column_widths = [85, 145, 90]  # Custom widths in pixels
        self.row_heights = [75, 90, 65]  # Custom heights in pixels

        self.frame1 = None
        self.frame2 = None
        self.processed_frame1 = None
        self.processed_frame2 = None

        self.reference_frame1 = None
        self.reference_frame2 = None

        self.ssim_values1 = [None] * 9
        self.ssim_values2 = [None] * 9

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
                for i in range(9):
                    y1, y2, x1, x2 = self.get_region_coords(i)
                    try:
                        self.ssim_values1[i] = ssim(self.reference_frame1[y1:y2, x1:x2],
                                                    self.processed_frame1[y1:y2, x1:x2])
                        self.ssim_values2[i] = ssim(self.reference_frame2[y1:y2, x1:x2],
                                                    self.processed_frame2[y1:y2, x1:x2])
                    except ValueError:
                        print(f"Error calculating SSIM for region {i}")
                        self.ssim_values1[i] = None
                        self.ssim_values2[i] = None

    def get_region_coords(self, region):
        # Calculate x coordinates based on column widths
        x1 = sum(self.column_widths[:region % 3])
        x2 = x1 + self.column_widths[region % 3]

        # Calculate y coordinates based on row heights
        y1 = sum(self.row_heights[:region // 3])
        y2 = y1 + self.row_heights[region // 3]

        return y1, y2, x1, x2

    def get_frames_with_ssim(self):
        frame1 = cv2.cvtColor(self.processed_frame1, cv2.COLOR_GRAY2BGR) if self.processed_frame1 is not None else None
        frame2 = cv2.cvtColor(self.processed_frame2, cv2.COLOR_GRAY2BGR) if self.processed_frame2 is not None else None

        if frame1 is not None and self.ssim_values1[0] is not None:
            frame1 = self.draw_regions_and_ssim(frame1, self.ssim_values1)

        if frame2 is not None and self.ssim_values2[0] is not None:
            frame2 = self.draw_regions_and_ssim(frame2, self.ssim_values2)

        return frame1, frame2

    def draw_regions_and_ssim(self, frame, ssim_values):
        h, w = frame.shape[:2]

        # Draw vertical region lines
        x = 0
        for width in self.column_widths[:-1]:
            x += width
            cv2.line(frame, (x, 0), (x, h), (0, 255, 0), 1)

        # Draw horizontal region lines
        y = 0
        for height in self.row_heights[:-1]:
            y += height
            cv2.line(frame, (0, y), (w, y), (0, 255, 0), 1)

        # Draw SSIM values for specific regions
        for region in [1, 3, 4, 5, 7]:
            y1, y2, x1, x2 = self.get_region_coords(region)
            if ssim_values[region] is not None:
                text = f"{ssim_values[region]:.2f}"
            else:
                text = "N/A"

            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Create a semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1 + 2, y1 + 2), (x1 + text_width + 8, y1 + text_height + 8), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.5, 0, frame)

            # Draw the text
            cv2.putText(frame, text, (x1 + 5, y1 + text_height + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

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

                if self.frame_count >= 60:
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

    def set_dimensions(self, column_widths, row_heights):
        with self.lock:
            self.column_widths = column_widths
            self.row_heights = row_heights


# Usage example
if __name__ == "__main__":
    processor = DualCameraProcessor(1, 0)  # Assuming camera indices 1 and 0
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