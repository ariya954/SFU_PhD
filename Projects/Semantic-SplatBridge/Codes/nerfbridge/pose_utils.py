import torch

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from scipy.spatial.transform import Rotation

"""
Utilities for converting ROS2 pose messages to torch tensors, and for converting
poses expressed in other coordinate systems to the Nerfstudio coordinate sytem.

Useful documentation:
    1. https://docs.nerf.studio/en/latest/quickstart/data_conventions.html
    2. https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/Calibration_Tutorial.pdf
"""

from typing import Union

def ros_msg_to_homogenous(pose_message: Union[Odometry, PoseStamped]):
    """
    Converts a ROS2 Pose message to a 4x4 homogenous transformation matrix
    as a torch tensor (half precision).
    """

    if isinstance(pose_message, PoseStamped):
        pose = pose_message.pose
    elif isinstance(pose_message, Odometry):
        pose = pose_message.pose.pose
    else:
        raise TypeError("pose_message must be of type PoseStamped or Odometry")

    quat = pose.orientation
    pose = pose.position

    R = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
    t = torch.Tensor([pose.x, pose.y, pose.z])

    T = torch.eye(4)
    T[:3, :3] = torch.from_numpy(R)
    T[:3, 3] = t
    return T.to(dtype=torch.float32)


def cuvslam_to_nerfstudio(T_cuvslam: torch.Tensor):
    """
    Converts a homogenous matrix 4x4 from the coordinate system used by the odometry topic
    in ISAAC ROS VSLAM to the Nerfstudio camera coordinate system 3x4 matrix.

    Equivalent operation to:
        T_out = T_in @ (Rotate Z by 180) @ (Rotate Y by 90) @ (Rotate Z by 90)
    """
    offset = torch.tensor([0.1, 0.03, 0.02, 1.0])
    T_cuvslam[:,-1] += offset

    T_ns = T_cuvslam[:, [1, 2, 0, 3]]
    T_ns[:, [0, 2]] *= -1
    return T_ns[:3, :]


def orbslam3_to_nerfstudio(T_orbslam3: torch.Tensor):
    """
    Converts a homogenous matrix 4x4 from the coordinate system used in orbslam3
    to the Nerfstudio camera coordinate system 3x4 matrix.

    Equivalent operation to:
        T_out = (Y by 180) @ (Z by 180) @ (X by 90) @ T_in @ (X @ 180)
    """
    T_ns = T_orbslam3[[0, 2, 1, 3], :]
    T_ns[:, [1, 2]] *= -1
    T_ns[2, :] *= -1
    return T_ns[:3, :]


def mocap_to_nerfstudio(T_mocap: torch.Tensor):
    """
    Converts a homogenous matrix 4x4 from the coordinate system used in mocap
    to the Nerfstudio camera coordinate system 3x4 matrix.
    """
    T_ns = T_mocap[:, [1, 2, 0, 3]]
    T_ns[:, [0, 2]] *= -1
    return T_ns[:3, :]


def get_homog_from_TFMsg(tfmsg):
    t = tfmsg.transform
    trans, rot = t.translation, t.rotation
    T = torch.eye(4)
    T[:3, :3] = torch.from_numpy(
        Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
    )
    T[:3, 3] = torch.tensor([trans.x, trans.y, trans.z])
    return T

def opencv_to_nerfstudio(T_cv):
    """
    Convert a camera pose from OpenCV format to NerfStudio format.
    OpenCV - x right, y down, and z forward
    NerfStudio - x right, y up, and z backward 
    180 degree rotation about the x-axis.
    """
    T_ns = T_cv
    T_ns[:, [1,2]] *= -1

    return T_ns[:3, :]

def slam_method_to_msgtype_func(slam_method: str):
    """
    Returns the message type that corresponds to the given SLAM method.
    """
    if slam_method == "cuvslam":
        return Odometry, cuvslam_to_nerfstudio
    elif slam_method == "orbslam3":
        return PoseStamped, orbslam3_to_nerfstudio
    elif slam_method == "tf":
        return Odometry, opencv_to_nerfstudio
    elif slam_method == "mocap":
        return PoseStamped, mocap_to_nerfstudio
    elif slam_method == "zedsdk":
        return PoseStamped, cuvslam_to_nerfstudio
    elif slam_method == "modal":
        return PoseStamped, opencv_to_nerfstudio
    else:
        raise NameError("Unknown SLAM Method!")
