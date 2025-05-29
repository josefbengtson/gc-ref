from pathlib import Path
import numpy as np
import kornia
import os
import time
import sys
import cv2
from PIL import Image

from RoMa_functions.DeDoDe.DeDoDe.utils import *
import poselib


def extract_normalized_translation(pose):
    t = pose[:3,3]
    t_normalized = t/np.linalg.norm(t)
    return t_normalized

def eval_pose_accuracy(P1, P2):
    # compute angle between t1 and t2
    t1 = P1[:3,3]
    t2 = P2[:3,3]
    t_angle_dist = np.arccos(np.dot(t1,t2)/(np.linalg.norm(t1)*np.linalg.norm(t2))) # Compute angle between t1 and t2

    t1_norm = extract_normalized_translation(P1)
    t2_norm = extract_normalized_translation(P2)
    t_dist = np.linalg.norm(t1_norm - t2_norm) # Compute normalized translation distance

    R1 = P1[:3,:3]
    R2 = P2[:3,:3]

    # Check R1 and R2 are orthogonal
    R_dot = np.dot(R1, R2.T)
    R_trace = np.trace(R_dot)
    R_arg = (R_trace-1)/2
    R_arg_clipped = np.clip(R_arg, -1, 1)
    rot_dist = np.arccos(R_arg_clipped) # Compute rotation distance between R1 and R2
    return t_dist, t_angle_dist, rot_dist


def match_kpts(im_A_path, im_B_path, detector, descriptor, matcher, threshold = 0.01):
    # Compute keypoints and descriptors for two images using DeDoDe
    num_keypoints = 10_000
    detections_A = detector.detect_from_path(im_A_path, num_keypoints = num_keypoints)
    keypoints_A, P_A = detections_A["keypoints"], detections_A["confidence"]
    detections_B = detector.detect_from_path(im_B_path, num_keypoints = num_keypoints)
    keypoints_B, P_B = detections_B["keypoints"], detections_B["confidence"]
    description_A = descriptor.describe_keypoints_from_path(im_A_path, keypoints_A)["descriptions"]
    description_B = descriptor.describe_keypoints_from_path(im_B_path, keypoints_B)["descriptions"]
    kpts_A, kpts_B, batch_ids = matcher.match(keypoints_A, description_A,
        keypoints_B, description_B,
        P_A = P_A, P_B = P_B,
        normalize = True, inv_temp=20, threshold = threshold) #Increasing threshold -> fewer matches, fewer outliers
    W_A, H_A = Image.open(im_A_path).size
    W_B, H_B = Image.open(im_B_path).size
    kpts_A, kpts_B = matcher.to_pixel_coords(kpts_A, kpts_B, H_A, W_A, H_B, W_B)
    kpts_A = kpts_A.cpu().numpy()
    kpts_B = kpts_B.cpu().numpy()
    return kpts_A, kpts_B

def poselib_pose_accuracy(P_gt, kpts_A, kpts_B,K, ransac_opt={}, bundle_opt={}, K2=None):
    focal_length = K[0, 0]
    principal_point = int(K[0, 2])
    camera = {'model': 'SIMPLE_PINHOLE', 'width': principal_point * 2, 'height': principal_point * 2,
              'params': [focal_length, principal_point, principal_point]}

    if K2 is None:
        K2 = K

    focal_length_2 = K2[0, 0]
    principal_point_2 = int(K2[0, 2])
    camera_2 = {'model': 'SIMPLE_PINHOLE', 'width': principal_point_2 * 2, 'height': principal_point_2 * 2,
                'params': [focal_length_2, principal_point_2, principal_point_2]}

    # Compute pose using poselib
    pose, info = poselib.estimate_relative_pose(kpts_A, kpts_B, camera, camera_2, ransac_opt, bundle_opt)


    np.set_printoptions(precision=2, suppress=True)
    torch.set_printoptions(precision=2, sci_mode=False)
    inliers = info['inliers']

    # Extract P_gen from estimated pose
    P_gen = np.eye(4)
    R_gen = pose.R
    assert np.allclose(R_gen.T, np.linalg.inv(R_gen)), "R_gen.T is not equal to R_gen.inv()"
    P_gen[:3, :3] = R_gen
    P_gen[:3, 3] = pose.t

    # Compute distances in degrees
    t_dist, t_angle_dist, rot_dist = eval_pose_accuracy(P_gt, P_gen)
    rot_dist_deg = np.rad2deg(rot_dist)
    t_angle_dist_deg = np.rad2deg(t_angle_dist)
    return P_gen, rot_dist_deg, t_angle_dist_deg, inliers
