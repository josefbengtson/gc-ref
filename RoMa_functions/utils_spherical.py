import torch
import torch.nn.functional as Fn
import numpy as np
import math
import random
import kornia
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



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


def compute_E(T1, T2):
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

    T_rel_pre = rel_T(T1, T2)

    # # TODO Testing Flip y axis

    # Convert into P1 = [I|0] and P2 = [R|t]
    print("------------------------- In Compute E -------------------------")
    print("T1: ", T1)
    print("T2: ", T2)
    T1_inv = torch.inverse(T1)
    print("T1_inv: ", T1_inv)
    T1_new = torch.matmul(T1, T1_inv)
    T2_new = torch.matmul(T2, T1_inv)


    print("T1_new: ", T1_new)
    print("T2_new: ", T2_new)
    T_rel = rel_T(T1, T2)
    T_rel_new = rel_T(T1_new, T2_new)

    assert ((T_rel-T_rel_new) < 1e-3).all(), "Relative pose has been changed!"

    # Compute Essential matrix
    R = T2_new[:3, :3]
    t = T2_new[:3, 3]
    print("R: ", R)
    print("t: ", t)
    t_x = skew_sym(t)
    print("t_x: ", t_x)
    E = t_x@R
    print("E: ", E)

    return E, T1_new, T2_new


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
        P1 = P1.double()
        P2 = P2.double()
        X = X.double()
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





def compute_F(K1, K2, T1, T2):
    # Compute fundamental matrix
    T1_kornia = torch.eye(4)[None].to(device)
    T1_kornia[0,:,:] = T1

    T2_kornia = torch.eye(4)[None].to(device)
    T2_kornia[0,:,:] = T2

    P1_kornia = T1_kornia[0,:3,:]
    P2_kornia = T2_kornia[0,:3,:]
    print("K1 device: ", K1.device)
    print("P1 kornia.device: ", P1_kornia.device)
    KP1_kornia = K1 @ P1_kornia.double()
    KP2_kornia = K2 @ P2_kornia.double()

    F = kornia.geometry.epipolar.fundamental_from_projections(KP1_kornia, KP2_kornia)
    return F

def compute_F_own(K1, K2, T1, T2):
    # Compute Essential matrix
    E, T1_new, T2_new = compute_E(T1, T2)
    # Compute Fundamental matrix
    # Print types of all inputs
    print("K1: ", K1.dtype)
    print("K2: ", K2.dtype)
    print("E: ", E.dtype)
    F = mult_three(torch.transpose(torch.inverse(K2),0,1),E,torch.inverse(K1))
    print("F before scaling: ", F)
    F = F/F[2,2] # Rescale so that F[2,2] = 1
    return F

