import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

from PIL import Image
import torch
import cv2 as cv
# from roma import roma_outdoor
# from RoMa.roma import roma_outdoor

import numpy as np
import kornia
from kornia.geometry.epipolar import get_closest_point_on_epipolar_line
import time
import random
import sys

# from RoMa_functions.pose_estimation import loss_consistency

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", DEVICE)
from RoMa_functions.utils_spherical import compute_F
# import utils_spherical
import torch.nn.functional
from RoMa_functions.utils_RoMa import roma_matching, downsample_good_samples, draw_matches, generate_distinct_colors, warp_images

def add_ones_transpose(kpts):
    ones_column = torch.ones(kpts.shape[0], 1).to(DEVICE)
    kpts_extended = torch.cat((kpts, ones_column), dim=1)
    return torch.transpose(kpts_extended,0,1)


def WelschLoss(distances, threshold):
    c = threshold/3
    rhos = 1-torch.exp(-0.5*(distances/c)**2)
    loss = torch.mean(rhos)
    return loss


def comp_SED(F, kpts1, kpts2):

    kpts1 = kpts1.double()
    kpts2 = kpts2.double()
 
    kpts1_batch = kpts1[None]
    kpts2_batch = kpts2[None]
    F = F[None]
    Ft = torch.transpose(F, 1, 2)

    # Compute epipolar lines
    e_line_im2 = kornia.geometry.epipolar.compute_correspond_epilines(kpts1_batch, F)
    e_line_im2 = e_line_im2.squeeze(0)

    e_line_im1 = kornia.geometry.epipolar.compute_correspond_epilines(kpts2_batch, Ft)
    e_line_im1 = e_line_im1.squeeze(0)

    # Get closest points on epipolar lines amd compute distances
    closest_point_im1 = get_closest_point_on_epipolar_line(kpts2_batch, kpts1_batch, Ft)
    diff_1 = (kpts1_batch - closest_point_im1).reshape(kpts1.shape[0], 2)
    distances_1 = torch.norm(diff_1, p=2, dim=1)

    closest_point_im2 = get_closest_point_on_epipolar_line(kpts1_batch, kpts2_batch, F)
    diff_2 = (kpts2_batch - closest_point_im2).reshape(kpts1.shape[0], 2)
    distances_2 = torch.norm(diff_2, p=2, dim=1)

    return distances_1, distances_2, e_line_im1, e_line_im2, closest_point_im1, closest_point_im2



def consistency_loss(params, F, cond_image, output_image):
    if output_image.requires_grad:
        output_image.retain_grad()  # Retain gradient for non-leaf tensor

    good_samples = params['good_samples']
    certainty_threshold = params['certainty_threshold']
    num_matches = params['num_matches']
    roma_model = params['roma_model']
    huber_loss = params['huber_loss']
    l1_loss = params['l1_loss']
    h = params['h']
    w = params['w']
    bCertaintySampling = params['bCertaintySampling']
    alpha_rgb = params['alpha_rgb']
    output_folder = params['output_folder']
    i_opt = params['i_opt']


    print("Num Matches in consistency loss: ", num_matches)
    if good_samples is None:
        print("Good samples is None")
    else:
        print("Length of good samples: ", len(good_samples))
        print("Number good samples: ", good_samples.sum())

    # good_samples = None # Ensure new good_samples chosen each time

    # Compute matching keypoints
    (kpts1, kpts2, certainty_map,
     good_samples, good_samples_all, matches) = roma_matching(
        cond_image, output_image, h, w, h, w, num_matches, certainty_threshold, DEVICE,
        roma_model=roma_model, good_samples=good_samples, match_based_on_certainty=bCertaintySampling)

    if len(kpts1.shape) == 3:
        kpts1 = kpts1.squeeze(0)
        kpts2 = kpts2.squeeze(0)

    print("kpts1 shape: ", kpts1.shape)
    print("kpts2 shape: ", kpts2.shape)
    num_points = kpts1.shape[0]
    bSaveKeypoints = False

    if bSaveKeypoints:
        current_azimuth = params["current_azimuth"]

        # Save keypoints to .npy
        kpts1_numpy = kpts1.cpu().detach().numpy()
        kpts2_numpy = kpts2.cpu().detach().numpy()
        if 'i_pose' in params.keys():
            # i_iter = params['i_pose']
            np.save(output_folder + f"/{current_azimuth}/analysis/keypoints/kpts1_{i_opt}.npy", kpts1_numpy)
            np.save(output_folder + f"/{current_azimuth}/analysis/keypoints/kpts2_{i_opt}.npy", kpts2_numpy)
        else:
            np.save(output_folder + f"/analysis/keypoints/kpts1_{i_opt}.npy", kpts1_numpy)
            np.save(output_folder + f"/analysis/keypoints/kpts2_{i_opt}.npy", kpts2_numpy)
    

    bSaveCertaintyMaps = True
    if bSaveCertaintyMaps:

        # Save image of certainty map
        certainty_map_numpy = certainty_map.cpu().detach().numpy()
        certainty_map_numpy = certainty_map_numpy.squeeze()
        certainty_map_save = Image.fromarray(255 * certainty_map_numpy).convert('RGB')
        certainty_map_reshaped = certainty_map_save.resize((256, 256))
        if 'i_pose' in params.keys():
            current_azimuth = params["current_azimuth"]

            # i_iter = params['i_pose']
            certainty_map_reshaped.save(output_folder + f"/{current_azimuth}/analysis/certainty_map_{i_opt}.png")
        else:
            certainty_map_reshaped.save(output_folder + f"/analysis/certainty_map_{i_opt}.png")
        print("certainty_map_save shape: ", certainty_map_save.size)


    bPerformWarping = True
    if bPerformWarping and num_points > 1:
        bStandardWarping = False
        if bStandardWarping:
            input_image_indices = torch.round(kpts1).long()
            output_image_indices = torch.round(kpts2).long()
            # Clamp input_image_indices and output_image_indices to valid image dimensions (h, w)
            input_image_indices[:, 1] = torch.clamp(input_image_indices[:, 1], min=0, max=h - 1)
            input_image_indices[:, 0] = torch.clamp(input_image_indices[:, 0], min=0, max=w - 1)
            output_image_indices[:, 1] = torch.clamp(output_image_indices[:, 1], min=0, max=h - 1)
            output_image_indices[:, 0] = torch.clamp(output_image_indices[:, 0], min=0, max=w - 1)


            device_cond_image = cond_image.device
            input_image_indices = input_image_indices.to(device_cond_image)
            extracted_values_input_image = cond_image[:, input_image_indices[:, 1], input_image_indices[:, 0]]
            device_output_image = output_image.device
            output_image_indices = output_image_indices.to(device_output_image)
            extracted_values_output_image = output_image[:, output_image_indices[:, 1],output_image_indices[:, 0]]
            extracted_values_input_image = extracted_values_input_image.to(DEVICE)
            extracted_values_output_image = extracted_values_output_image.to(DEVICE)
            loss_rgb = l1_loss(extracted_values_input_image, extracted_values_output_image)
        # print("Loss RGB: ", loss_rgb)
        else:
            # New approach
            kpts1_normalized = matches[:, :2]
            kpts2_normalized = matches[:, 2:]
            device_cond_image = cond_image.device
            kpts1_normalized = kpts1_normalized.to(device_cond_image)
            # Sample values from the input and output images using grid_sample
            extracted_values_input_image_new = torch.nn.functional.grid_sample(cond_image[None], kpts1_normalized[None][None],
                                                         align_corners=False).squeeze(0)
            device_output_image = output_image.device
            kpts2_normalized = kpts2_normalized.to(device_output_image)
            extracted_values_output_image_new = torch.nn.functional.grid_sample(output_image[None], kpts2_normalized[None][None],
                                                          align_corners=False).squeeze(0)


            extracted_values_input_image_new = extracted_values_input_image_new.to(DEVICE)
            extracted_values_output_image_new = extracted_values_output_image_new.to(DEVICE)

            loss_rgb = l1_loss(extracted_values_input_image_new, extracted_values_output_image_new)



    else:
        loss_rgb = None


    # Compute epipolar loss
    if num_points > 1:
        ED1, ED2, e_line_im_1, e_line_im_2, closest_point_im1, closest_point_im2 = comp_SED(F, kpts1, kpts2) # Compute epipolar distances

        SEDs = (ED1 + ED2) / 2

        zero_tensor = torch.zeros(SEDs.shape, dtype=torch.double).to(DEVICE)

        loss_consistency = huber_loss(SEDs, zero_tensor)

    else:
        ED1, ED2, e_line_im_1, e_line_im_2, closest_point_im1, closest_point_im2 = comp_SED(F, kpts1, kpts2) # Compute epipolar distances
        zero_tensor = torch.zeros(ED1.shape, dtype=torch.double).to(DEVICE)
        SEDs = (ED1 + ED2) / 2
        inlier_threshold_huber_loss = 2
        huber_loss_sum = torch.nn.HuberLoss(delta=inlier_threshold_huber_loss, reduction='sum')

        loss_consistency = huber_loss_sum(SEDs, zero_tensor)
        SEDs = None
        e_line_im_1 = None
        e_line_im_2 = None
        closest_point_im1 = None
        closest_point_im2 = None


    alpha_consistency = 1
    if loss_rgb is not None:
        print("Using RGB L1 Loss")
        print("alpha_rgb for l1 loss: ", alpha_rgb)
        print("alpha_consistency for consistency loss: ", alpha_consistency)
        loss = alpha_consistency*loss_consistency + alpha_rgb * loss_rgb
    else:
        print("Not Using RGB L1 Loss")
        loss = loss_consistency

    print("Using RGB L1 Loss: ", loss_rgb is not None)
    # Create output dict
    output_dict = {
        "loss": loss,
        "loss_consistency": loss_consistency,
        "loss_rgb": loss_rgb,
        "good_samples": good_samples,
        "SEDs": SEDs,
        "ED1": ED1,
        "ED2": ED2,
        "e_line_im_1": e_line_im_1,
        "e_line_im_2": e_line_im_2,
        "closest_point_im1": closest_point_im1,
        "closest_point_im2": closest_point_im2,
        "kpts1": kpts1,
        "kpts2": kpts2
    }
    return loss, output_dict
