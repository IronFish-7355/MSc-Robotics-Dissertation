import os
from PIL import Image


def crop_images(input_path, output_path, crop_dimensions):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get a list of all image files in the input directory
    image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    for image_file in image_files:
        # Open the image
        with Image.open(os.path.join(input_path, image_file)) as img:
            # Crop the image
            cropped_img = img.crop(crop_dimensions)

            # Save the cropped image
            output_file = os.path.join(output_path, f"cropped_{image_file}")
            cropped_img.save(output_file)
            print(f"Cropped and saved: {output_file}")


# Set your input and output paths
input_path = r"C:\Users\Hongg\Pictures\Normal_system"
output_path = r"C:\Users\Hongg\Pictures\Normal_system\Cropped"

# Set your crop dimensions (left, top, right, bottom)
crop_dimensions = (13,980,4784,3000)  # This will crop a 400x400 pixel area

# Call the function to crop the images
crop_images(input_path, output_path, crop_dimensions)