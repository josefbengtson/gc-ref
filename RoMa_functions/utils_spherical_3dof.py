import torch
import torch.nn.functional as Fn
import numpy as np
import math
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def camera_position_from_spherical_angles(
    r: float,
    elevation: float,
    azimuth: float,
    degrees: bool = True,
) -> torch.Tensor:
    """
    Calculate the location of the camera based on the distance away from
    the target point, the elevation and azimuth angles.

    Args:
        distance: distance of the camera from the object.
        elevation, azimuth: angles.
        degrees: bool, whether the angles are specified in degrees or radians.
        device: str or torch.device, device for new tensors to be placed on.

    Returns:
        camera_position: (N, 3) xyz location of the camera.
    """
    # If degrees convert to radians
    if degrees:
        elevation = math.pi / 180.0 * elevation
        azimuth = math.pi / 180.0 * azimuth
    elevation = torch.tensor(elevation)
    azimuth = torch.tensor(azimuth)

    # ORIGINAL Compute 3D position of camera center ORIGINAL
    x = r * torch.sin(elevation) * torch.cos(azimuth)
    y = r * torch.sin(elevation) * torch.sin(azimuth)
    z = r * torch.cos(elevation)

    # ZERONVS ADAPTED Compute 3D position of camera center
    # x = r * torch.cos(elevation) * torch.cos(azimuth)
    # y = r * torch.cos(elevation) * torch.sin(azimuth)
    # z = r * torch.sin(elevation)

    camera_position = torch.stack([x, y, z])
    return camera_position.view(-1, 3)


def look_at_rotation(
    camera_position, at=((0, 0, 0),), up=((0, 1, 0),)) -> torch.Tensor:
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.

    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.

    Args:
        camera_position: position of the camera in world coordinates
        at: position of the object in world coordinates
        up: vector specifying the up direction in the world coordinate frame.



    The vectors are broadcast against each other so they all have shape (N, 3).

    Returns:
        R: (3, 3) rotation matrix
    """

    at = torch.tensor(at, dtype=torch.float32)
    up = torch.tensor(up, dtype=torch.float32)
    # print("In look_at_rotation")
    # print("camera_position: ", camera_position)
    # print("at: ", at)
    # print("up: ", up)
    # ORIGINAL Compute the axis of rotation matrix
    z_axis = Fn.normalize(at - camera_position, eps=1e-5)
    x_axis = Fn.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = Fn.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)

    # ZERONVS ADAPTED Compute the axis of rotation matrix
    # z_axis = Fn.normalize(at - camera_position, eps=1e-5)
    # x_axis = Fn.normalize(torch.cross(z_axis, up, dim=1), eps=1e-5)
    # y_axis = Fn.normalize(torch.cross(x_axis, z_axis, dim=1), eps=1e-5)

    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(
        dim=1, keepdim=True
    )
    if is_close.any():
        replacement = Fn.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], -z_axis[:, None, :]), dim=1)
    return torch.reshape(R,[3,3])


def gen_T(elevation, azimuth, r, up=(0, 0, 1)):
    """
    This function generates a Transformation Matrix from a 3DoF position.

    The output is a transformation matrix

    Args:
        elevation: Elevation angle
        azimuth: Azimuth angle
        r: Radius
        up: Up direction

    Returns:
        T: Generated Transformation Matrix

    """
    # Compute camera center

    C = camera_position_from_spherical_angles(r, elevation, azimuth)
    # Compute look at rotation matrix
    R = look_at_rotation(C, up=(up,))
    # Compute translation t
    t = -torch.matmul(R, C[:, :, None])[:, :, 0]
    # Combine into transformation matrix T
    T = torch.zeros((4, 4))
    T[3, 3] = 1
    T[:3, :3] = R
    T[:3, 3] = t
    flip_z = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ], dtype=T.dtype)
    T = torch.matmul(flip_z, T)
    # T = torch.linalg.inv(T)
    return T


def print_array(T,name,decimals = 3):
    print(f"{name}: ", np.round(np.array(T),decimals))
    print("\n")


def rel_T(T1,T2):
    T2_inv = torch.inverse(T2)
    T_rel = torch.matmul(T1,T2_inv)
    return T_rel


def mult_three(T1,T2,T3):
    T_new = torch.matmul(T1,torch.matmul(T2,T3))
    return T_new


def skew_sym(t):
    t_x = torch.tensor([[0, -t[2], t[1]],
                       [t[2], 0, -t[0]],
                       [-t[1], t[0], 0]])
    return t_x


def camera_center(T):
    R_T = torch.transpose(T[:3, :3],0,1)
    t = T[:3, 3]
    C = - torch.matmul(R_T,t)
    return C


def compute_E(elevation_1,azimuth_1,r_1,elevation_2,azimuth_2,r_2, up=(0, 0, 1)):
    """
    This function 3DoF positions of two cameras and computes there essential matrix E.

    The output is an Essential Matrix.

    Args:
        elevation_1, elevation_2: Elevation angle for each of the two cameras
        azimuth_1, azimuth_2: Azimuth angle for each of the two cameras
        r_1, r_2: Radius for each of the two cameras

    Returns:
        E: (3, 3) Essential Matrix
        T1, T2: Original Transformation Matrices
        T1_new, T2_new: Relative Transformation Matrices

    """
    T1 = gen_T(elevation_1, azimuth_1, r_1, up)
    T2 = gen_T(elevation_2, azimuth_2, r_2, up)
    T_rel_pre = rel_T(T1, T2)

    # Flip the y axis
    flip_y = torch.tensor([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=T1.dtype)

    T1 = torch.matmul(flip_y, T1)
    T2 = torch.matmul(flip_y, T2)

    # Convert into P1 = [I|0] and P2 = [R|t]
    T1_inv = torch.inverse(T1)
    T1_new = torch.matmul(T1, T1_inv)
    T2_new = torch.matmul(T2, T1_inv)
    T_rel = rel_T(T1, T2)
    T_rel_new = rel_T(T1_new, T2_new)

    assert ((T_rel-T_rel_new) < 1e-3).all(), "Relative pose has been changed!"

    # Compute Essential matrix
    R = T2_new[:3, :3]
    t = T2_new[:3, 3]
    t_x = skew_sym(t)
    E = torch.matmul(t_x,R)


    return E, T1, T2, T1_new, T2_new


def Test_E(E, T1, T2, num_iter=1000):
    # Test if E condition holds for a randomly generated 3D points
    count_fails = 0
    check_E_max = 0
    for i in range(num_iter):
        X = torch.rand([4, 1]) * 10
        X[3] = 1
        # print("X: ", X)
        P1 = T1[:3, :4]
        P2 = T2[:3, :4]

        x1 = torch.matmul(P1, X)
        x2 = torch.matmul(P2, X)
        x1 = x1 / x1[2]
        x2 = x2 / x2[2]
        check_E = mult_three(torch.transpose(x2, 0, 1), E, x1).item()

        if check_E > 1e-4:
            if check_E > check_E_max:
                check_E_max = check_E
            count_fails = count_fails + 1
    return check_E_max, count_fails


def generate_random_values():
    elevation = random.uniform(0, 360)
    azimuth = random.uniform(0, 360)
    r = random.uniform(0.5, 5)
    return elevation, azimuth, r


def evaluate_many_E(num_iter = 100):
    average_largest = 0
    average_fails = 0
    for j in range(num_iter):
        # Generate random values for the first set
        elevation_1, azimuth_1, r_1 = generate_random_values()

        # Generate random values for the second set
        elevation_2, azimuth_2, r_2 = generate_random_values()
        E, T1, T2, T1_new, T2_new = compute_E(elevation_1,azimuth_1,r_1,elevation_2,azimuth_2,r_2)

    #     print("E: ", E)
        check_E_max, count_fails = Test_E(E, T1, T2)
        average_largest = average_largest + check_E_max
        average_fails = average_fails + count_fails
#         i
    average_largest = average_largest/num_iter
    average_fails = average_fails/num_iter
    return average_largest, average_fails


def compute_F(K1, K2, elevation_1,azimuth_1,r_1,elevation_2,azimuth_2,r_2, up):
    # Compute Essential matrix
    E, T1, T2, T1_new, T2_new = compute_E(elevation_1,azimuth_1,r_1,elevation_2,azimuth_2,r_2, up)

    # Compute Fundamental matrix
    F = mult_three(torch.transpose(torch.inverse(K2),0,1),E,torch.inverse(K1))
    F = F/F[2,2] # Rescale so that F[2,2] = 1
    return F


def compute_E_6dof(P1, P2):
    """
    This function 3DoF positions of two cameras and computes there essential matrix E.

    The output is an Essential Matrix.

    Args:
        elevation_1, elevation_2: Elevation angle for each of the two cameras
        azimuth_1, azimuth_2: Azimuth angle for each of the two cameras
        r_1, r_2: Radius for each of the two cameras

    Returns:
        E: (3, 3) Essential Matrix
        T1, T2: Original Transformation Matrices
        T1_new, T2_new: Relative Transformation Matrices

    """
    T1 = torch.eye(4)
    T1[:3, :] = P1
    # T1 = torch.inverse(T1)

    T2 = torch.eye(4)
    T2[:3, :] = P2
    # T2 = torch.inverse(T2)

    T1_inv = torch.inverse(T1)
    T1_new = torch.matmul(T1, T1_inv)
    T2_new = torch.matmul(T2, T1_inv)

    # T1_new = torch.matmul(T1_inv, T1)
    # T2_new = torch.matmul(T1_inv, T2)

    # T2_new = torch.inverse(T2_new)

    assert ((T1_new[:3, :] - torch.eye(3, 4)) < 1e-5).all(), "P1 not identity!"

    P2_new = T2_new[:3, :]

    T_rel = rel_T(T1, T2)
    T_rel_new = rel_T(T1_new, T2_new)

    assert ((T_rel - T_rel_new) < 1e-3).all(), "Relative pose has been changed!"

    R = T2_new[:3, :3]
    t = T2_new[:3, 3]

    t_x = skew_sym(t)
    E = torch.matmul(t_x, R)

    return E, P2_new


def compute_F_6dof(K, P1, P2):
    E, P2_new = compute_E_6dof(P1,P2)
    K1, K2 = K, K
    # F = mult_three(torch.transpose(torch.inverse(K),0,1),E,torch.inverse(K))
    F = K2.inverse().transpose(-2, -1) @ E @ K1.inverse()
    F = F/F[2,2]
    return F, P2_new
