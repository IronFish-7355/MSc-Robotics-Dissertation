import numpy as np
from scipy.spatial.transform import Rotation as R


def create_transform(x, y, z, rx, ry, rz):
    """Create a 4x4 homogeneous transformation matrix from position and euler angles."""
    rot = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
    trans = np.array([x, y, z])
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans
    return transform


def solve_ax_xb(A, B):
    """Solve AX=XB equation for X."""
    # Extract rotation matrices
    RA = A[:3, :3]
    RB = B[:3, :3]

    # Solve for rotation
    M = np.dot(RA, RB.T)
    eigenvalues, eigenvectors = np.linalg.eig(M)

    # Find the eigenvector corresponding to the eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1))

    # Convert the eigenvector to a proper rotation matrix
    u = eigenvectors[:, idx].real
    u = u / np.linalg.norm(u)  # Normalize the vector

    # Construct a rotation matrix using the Rodriguez formula
    theta = np.arccos((np.trace(M) - 1) / 2)
    K = np.array([
        [0, -u[2], u[1]],
        [u[2], 0, -u[0]],
        [-u[1], u[0], 0]
    ])
    RX = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    # Solve for translation
    I = np.eye(3)
    tA = A[:3, 3]
    tB = B[:3, 3]
    tX = np.dot(np.linalg.inv(I - RA), tA - np.dot(RX, tB))

    # Construct the transformation matrix
    X = np.eye(4)
    X[:3, :3] = RX
    X[:3, 3] = tX

    return X


def eye_to_hand_calibration(robot_pose, marker_pose):
    """Perform eye-to-hand calibration using a single data point."""
    # A: robot end-effector pose in robot base frame
    # B: marker pose in camera frame
    # X: camera to robot base transformation
    # We want to solve: AX = XB

    X = solve_ax_xb(robot_pose, marker_pose)
    return X


# Example usage
if __name__ == "__main__":
    # Sample data (using only one data point)
    robot_poses_data = [
        -12.581, -4.607, 407.354, -0, 0.785398,
         -1.5707]

    camera_poses_data = [-28, -306, 613, 2.672, 0.2075,
         0.61777]

    # Convert poses to transformation matrices
    robot_pose = create_transform(*robot_poses_data)
    print("robot_pose")
    print(robot_pose)
    marker_pose = create_transform(*camera_poses_data)
    print("camera_pose")
    print(marker_pose)
    # Perform eye-to-hand calibration
    T_base_camera = eye_to_hand_calibration(robot_pose, marker_pose)

    print("Calibration result (transformation from camera frame to robot base frame):")
    print("T_base_camera = np.array([")
    for row in T_base_camera:
        print(f"    [{', '.join([f'{x:.8e}' for x in row])}],")
    print("])")