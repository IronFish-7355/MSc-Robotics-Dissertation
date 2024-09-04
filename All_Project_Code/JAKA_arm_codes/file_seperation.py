import csv
import os
import shutil

# Define paths
csv_path = r"C:\Users\Hongg\Desktop\JAKA_arm_codes_backup_20240807_afternoon\DATA\proper_grasps_20240827_140834.csv"
video_dir = r"C:\Users\Hongg\Desktop\JAKA_arm_codes_backup_20240807_afternoon\DATA\0"
target_dir = r"C:\Users\Hongg\Desktop\JAKA_arm_codes_backup_20240807_afternoon\DATA\1"


def read_csv(file_path):
    proper_grasps = set()
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            z, y, ry = map(float, row)
            proper_grasps.add((round(z), round(y), round(ry)))
    return proper_grasps


def format_offset(value):
    return f"+{value}" if value >= 0 else str(value)


def main():
    proper_grasps = read_csv(csv_path)

    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    for filename in os.listdir(video_dir):
        if filename.startswith('Z') and filename.endswith('.mp4'):
            # Extract offsets from filename
            parts = filename[:-4].split('Y')
            z = int(parts[0][1:])
            y, ry = map(int, parts[1].split('ry'))

            if (z, y, ry) in proper_grasps:
                source = os.path.join(video_dir, filename)
                destination = os.path.join(target_dir, filename)
                shutil.move(source, destination)
                print(f"Moved {filename} to {target_dir}")


if __name__ == "__main__":
    main()