import pyrealsense2 as rs
import numpy as np
import cv2


def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align


def process_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None, None

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return depth_frame, depth_image, color_image


def detect_objects(color_image, depth_frame):
    # Convert to LAB color space
    lab_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)

    # Define color range for detection (adjust as needed)
    lower_lab = np.array([0, 149, 127])
    upper_lab = np.array([255, 255, 255])

    # Create mask and find contours
    mask = cv2.inRange(lab_image, lower_lab, upper_lab)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_objects = []
    for contour in contours:
        if cv2.contourArea(contour) > 150:  # Adjust this threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w // 2, y + h // 2
            depth = depth_frame.get_distance(center_x, center_y)
            detected_objects.append(((center_x, center_y), depth, (x, y, w, h)))

    return detected_objects, mask


def display_results(color_image, segmented_image, detected_objects):
    # Create a copy of the color image for drawing
    display_image = color_image.copy()

    # Draw detected objects on the display image
    for i, (center, depth, bbox) in enumerate(detected_objects):
        x, y, w, h = bbox
        cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(display_image, center, 5, (0, 255, 0), -1)
        cv2.putText(display_image, f"Obj {i + 1}, Depth: {depth:.2f}m", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert binary segmented image to BGR for colormap application
    segmented_color = cv2.applyColorMap(segmented_image, cv2.COLORMAP_JET)

    # Combine the two images side by side
    combined_image = np.hstack((display_image, segmented_color))

    # Display the combined image
    cv2.imshow("RGB with Detection | Segmented Image", combined_image)


def main():
    pipeline, align = initialize_realsense()

    try:
        while True:
            depth_frame, depth_image, color_image = process_frames(pipeline, align)
            if depth_frame is None:
                continue

            detected_objects, segmented_image = detect_objects(color_image, depth_frame)
            display_results(color_image, segmented_image, detected_objects)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()