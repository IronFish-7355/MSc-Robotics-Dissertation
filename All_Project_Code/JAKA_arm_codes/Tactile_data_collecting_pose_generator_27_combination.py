import math

# Given offset values
Tac_Z_offset = 10  # mm
Tac_Y_offset = 10  # mm

# Accurate object pose
Object_Accu_pose_space = [-364.378, 357.315, 238.233, 0, 0, -3.141589771113383]

# Initialize the list to hold all 9 poses
Object_offset_poses = []

# Generate all combinations (+, 0, -) for Z and Y
    for y_offset in [Tac_Y_offset, 0, -Tac_Y_offset]:
        # Copy the original pose
        pose = Object_Accu_pose_space.copy()

        # Adjust Y (2nd element)
        pose[1] += y_offset
        # Add the modified pose to the list
        Object_offset_poses.append(pose)

# Assign to respective variables
Object_offset_pose_1 = Object_offset_poses[0]  # Z +, Y +
Object_offset_pose_2 = Object_offset_poses[1]  # Z +, Y 0
Object_offset_pose_3 = Object_offset_poses[2]  # Z +, Y -



# For clarity, let's print all 9 poses
for i in range(1, 10):
    print(f"Object_offset_pose_{i} =", eval(f"Object_offset_pose_{i}"))

Object_offset_pose_1 = [-364.378, 367.315, 248.233, 0, 0, -3.141589771113383]  # Z +, Y +
Object_offset_pose_2 = [-364.378, 357.315, 248.233, 0, 0, -3.141589771113383]  # Z +, Y 0
Object_offset_pose_3 = [-364.378, 347.315, 248.233, 0, 0, -3.141589771113383]  # Z +, Y -
Object_offset_pose_4 = [-364.378, 367.315, 238.233, 0, 0, -3.141589771113383]  # Z 0, Y +
Object_offset_pose_5 = [-364.378, 357.315, 238.233, 0, 0, -3.141589771113383]  # Z 0, Y 0
Object_offset_pose_6 = [-364.378, 347.315, 238.233, 0, 0, -3.141589771113383]  # Z 0, Y -
Object_offset_pose_7 = [-364.378, 367.315, 228.233, 0, 0, -3.141589771113383]  # Z -, Y +
Object_offset_pose_8 = [-364.378, 357.315, 228.233, 0, 0, -3.141589771113383]  # Z -, Y 0
Object_offset_pose_9 = [-364.378, 347.315, 228.233, 0, 0, -3.141589771113383]  # Z -, Y -
