import torch
import sys
from RoMa_functions.RoMa.roma import roma_outdoor
import torchvision.transforms.functional as tvF
import time
import numpy as np
from PIL import Image, ImageDraw
import os
import wandb
import cv2
import matplotlib.colors as mcolors
import random

# from model_zoo import roma_outdoor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def numpy_to_pil(x: np.ndarray):
    """
    Args:
        x: Assumed to be of shape (h,w,c)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.max() <= 1.01:
        x *= 255
    x = x.astype(np.uint8)
    return Image.fromarray(x)

def tensor_to_pil(x):
    assert x.is_cuda, "Tensor is not on CUDA"
    assert x.dim() == 3, "Tensor must be 3-dimensional"
    assert x.size(1) > 0 and x.size(2) > 0, "Tensor dimensions must be positive"

    x = x.detach().permute(1, 2, 0).cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return numpy_to_pil(x)



def draw_matches(kpts1, kpts2, input_im_path, output_im_path, num_matches_show, draw_lines=True,
                 e_line_im_1=None, closest_point_im1=None, e_line_im_2=None, closest_point_im2=None, random_sampling=True, SEDs=None):
    if draw_lines:
        circle_size = 1
    else:
        circle_size = 3
    img1 = cv2.imread(input_im_path)  # queryImage
    img1 = img1[:, :, ::-1].copy()
    img2 = cv2.imread(output_im_path)  # trainImage
    img2 = img2[:, :, ::-1].copy()

    num_total = kpts1.shape[0]
    if random_sampling:
        if num_total < num_matches_show:
            num_matches_show = num_total
        indices = random.sample(range(num_total), num_matches_show)
    else:
        indices = np.linspace(0, num_total-1, num_matches_show, dtype=int)
    # print("indices: ", indices)
    if num_matches_show<50:
        colors_values = generate_distinct_colors(num_matches_show)
    else:
        colors_values = [(0, 0, 255) for i in range(num_matches_show)]

    combined_img = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    color_value_index = 0
    for i in indices:
        color_value = colors_values[color_value_index]
        color_value_index += 1
        x1, y1 = map(int, kpts1[i])
        cv2.circle(img1, (x1, y1), circle_size, color_value, -1)  # Draw blue circles for points

        x2, y2 = map(int, kpts2[i])
        # if draw_lines:
        cv2.circle(img2, (x2, y2), circle_size, color_value, -1)  # Draw blue circles for points
        if e_line_im_2 is not None:
            #print("e_line_im_2 shape: ", e_line_im_2.shape)
            #print(" e_line_im_2[i, :] shape: ",  e_line_im_2[i, :].shape)
            a, b, c = e_line_im_2[i, :]
            #print("a: ", a, ", b: ", b, ", c: ", c)
            lx0, ly0 = map(int, [0, -c/b])
            lx1, ly1 = map(int, [img2.shape[1]-1, -(c + a*(img2.shape[1]-1))/b])
            cv2.line(img2, (lx0, ly0), (lx1, ly1), color_value, 1)  # Draw green epipolar lines
            if SEDs is not None:
                cv2.putText(img2, "%.1f" % SEDs[i], (int(x2) + 5, int(y2) + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, color_value, 1, cv2.LINE_AA)

            if closest_point_im2 is not None:
                x_line, y_line = map(int, closest_point_im2[0, i])
                # cv2.circle(img2, (x_line, y_line), 3, (0, 0, 255), -1)  # Draw red circles for points
                cv2.line(img2, (x2, y2), (x_line, y_line), (0, 255, 255), 1)  # Draw cyan line between points

        if e_line_im_1 is not None:
            #print("e_line_im_1 shape: ", e_line_im_1.shape)
            #print(" e_line_im_1[i, :] shape: ",  e_line_im_1[i, :].shape)
            a, b, c = e_line_im_1[i, :]
            #print("a: ", a, ", b: ", b, ", c: ", c)
            lx0, ly0 = map(int, [0, -c/b])
            lx1, ly1 = map(int, [img1.shape[1]-1, -(c + a*(img1.shape[1]-1))/b])
            cv2.line(img1, (lx0, ly0), (lx1, ly1), color_value, 1)  # Draw green epipolar lines

            if closest_point_im1 is not None:
                x_line, y_line = map(int, closest_point_im1[0, i])
                # cv2.circle(img2, (x_line, y_line), 3, (0, 0, 255), -1)  # Draw red circles for points
                cv2.line(img1, (x1, y1), (x_line, y_line), (0, 255, 255), 1)  # Draw cyan line between points

    # Copy img1 and img2 onto the combined image
    combined_img[:img1.shape[0], :img1.shape[1]] = img1
    combined_img[:img2.shape[0], img1.shape[1]:] = img2
    combined_img[:, img1.shape[1]:img1.shape[1] + 3] = (0, 0, 0)  # Draw a black vertical line

    if draw_lines:
        color_value_index = 0
        for i in indices:
            # Image 1

            color_value = colors_values[color_value_index]
            color_value_index += 1
            x1, y1 = map(int, kpts1[i])
            x2, y2 = map(int, kpts2[i])

            # Draw lines between corresponding points in img1 and epipolar lines in img2 on the combined image
            cv2.line(combined_img, (x1, y1), (x2 + img1.shape[1], y2), color_value, 1)

    combined_img_PIL = Image.fromarray(combined_img)
    return combined_img_PIL

def generate_distinct_colors(N,rescale=False):
    """
    Generate N distinct RGB triplets.

    Args:
    N (int): Number of distinct colors to generate.

    Returns:
    list: List of RGB triplets.
    """
    # Generate N distinct colors in HSV space
    hsv_colors = [(i / N, 1.0, 1.0) for i in range(N)]

    # Convert HSV colors to RGB
    rgb_colors = [mcolors.hsv_to_rgb(hsv) for hsv in hsv_colors]

    if not rescale:
        # Convert to 8-bit RGB values
        rgb_colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in rgb_colors]

    return rgb_colors




def downsample_good_samples_random(good_samples, target_count=5000):
    # Find the indices of the True values
    true_indices = torch.nonzero(good_samples, as_tuple=True)[0]

    # If there are more than target_count True values, randomly select target_count of them
    if len(true_indices) > target_count:
        true_indices = true_indices[torch.randperm(len(true_indices))[:target_count]]

    # Create a new array of False values
    downsampled = torch.zeros_like(good_samples, dtype=torch.bool)

    # Set the selected indices to True
    downsampled[true_indices] = True

    return downsampled

def downsample_good_samples(good_samples, target_count=5000):
    # Find the indices of the True values
    true_indices = torch.nonzero(good_samples, as_tuple=True)[0]
    print("In downsample_good_samples")
    print("Number true indices: ", len(true_indices))
    print("True indices: ", true_indices)
    print("Target count: ", target_count)
    # Calculate the step size to evenly distribute the True values
    step_size = max(1, len(true_indices) // target_count)
    print("Step size: ", step_size)
    print("Max index in true indices: ", true_indices[-1])
    # Select indices based on the calculated step size
    selected_indices = true_indices[::step_size]
    # If selected_indices has more than target_count elements, sample evenly
    if len(selected_indices) > target_count:
        # Generate evenly spaced indices
        evenly_spaced_indices = torch.linspace(0, len(selected_indices) - 1, target_count, dtype=torch.long)
        # Use these indices to sample from selected_indices
        selected_indices = selected_indices[evenly_spaced_indices]
    print("Number selected indices: ", len(selected_indices))
    print("Max selected index: ", selected_indices[-1])
    print("Selected indices: ", selected_indices)
    # Create a new array of False values
    downsampled = torch.zeros_like(good_samples, dtype=torch.bool)

    # Set the selected indices to True
    downsampled[selected_indices] = True
    print("Number true in downsampled: ", downsampled.sum())
    return downsampled


def roma_matching(im1, im2, H_A, W_A, H_B, W_B, num_matches, certainty_threshold,
                  device, roma_model=None, good_samples=None, match_based_on_certainty=False, bFilterMatches=True):
    if roma_model is None:
        print("--------RomA Model not provided, creating new model--------")
        roma_model = roma_outdoor(device=device)
        roma_model.decoder.detach = False
        roma_model.upsample_preds = False

    if len(im1.shape) == 3:
        im1 = im1.unsqueeze(0)
    elif not (len(im1.shape) == 4 and im1.shape[0] == 1):
        raise ValueError(f"Expected tensor shape [1,H,W,3], but got {im1.shape}")

    if len(im2.shape) == 3:
        im2 = im2.unsqueeze(0)
    elif not (len(im2.shape) == 4 and im2.shape[0] == 1):
        raise ValueError(f"Expected tensor shape [1,H,W,3], but got {im2.shape}")    


    H, W = 560, 560
    print("RoMa dimensions check")
    print("im 1 shape: ", im1.shape)
    print("im2 shape: ", im2.shape)
    print("H: ", H)
    print("W: ", W)
    im1 = tvF.resize(im1, (H, W))
    im2 = tvF.resize(im2, (H, W))


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im1 = tvF.normalize(im1, mean=mean, std=std)
    im2 = tvF.normalize(im2, mean=mean, std=std)

    warp, certainty_map = roma_model.match(im1, im2, device=device, batched=True)


    matches = warp.reshape(-1, 4)


    good_samples_all = None
    if good_samples is None:
        print("No good samples provided, using certainty map to filter matches")

        certainty_map_sample = certainty_map.clone()
        certainty_map_flat = certainty_map_sample.reshape(-1)
        if match_based_on_certainty:
            bTopK = False
            if not bTopK:
                # use samples over threshold
                good_samples_all = certainty_map_flat > certainty_threshold
                print("Number of good samples after certainty filtering: ", good_samples_all.sum())
            else:
                # Use top samples
                # Get the indices of the 1000 highest values
                num_include = int(certainty_threshold)
                print("Number of top matches to include: ", num_include)
                topk_values, topk_indices = torch.topk(certainty_map_flat, num_include)
                # Create a boolean mask with the same shape as certainty_map_flat
                good_samples_all = torch.zeros_like(certainty_map_flat, dtype=torch.bool)
                # Set the top 1000 indices to True
                good_samples_all[topk_indices] = True


            if num_matches < good_samples_all.sum():
                good_samples = downsample_good_samples(good_samples_all, num_matches)
            else:
                good_samples = good_samples_all
            # good_samples = good_samples_all
        else:
            # Down sample good_samples so only num_matches are used
            good_samples = downsample_good_samples(certainty_map_flat, num_matches)


    else:
        print("Good samples provided, using them to filter matches")
    print("Number of matches after downsampling: ", good_samples.sum())
    # if bFilterMatches:
    matches = matches[good_samples]


    # Convert to pixel coordinates
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)


    assert kpts1.shape[0] == kpts2.shape[0], "Number of keypoints do not match"

    return kpts1, kpts2, certainty_map, good_samples, good_samples_all, matches


def warp_images(kpts1_input, kpts2_input, cond_image_input_warping, output_tensor_input_warping, h, cond_folder_name,
                target_folder_name, output_folder, i_opt):
    time_pre_warping = time.time()
    if len(cond_image_input_warping.shape) == 4:
        cond_image_input_warping = cond_image_input_warping.squeeze(0)

    cond_image_warped = torch.zeros_like(cond_image_input_warping)
    output_image_warped = torch.zeros_like(output_tensor_input_warping)
        
    print("cond_image_warped shape: ", cond_image_warped.shape)
    print("output_image_warped shape: ", output_image_warped.shape)
    print("kpts1_all[:, 0].max()")
    # Convert coordinates to integers and handle out-of-bounds
    # print("h: ", h)
    kpts1_all = kpts1_input.clamp(0, h - 1).long()
    kpts2_all = kpts2_input.clamp(0, h - 1).long()

    # Ensure indices are within bounds
    assert kpts1_all[:, 0].max() < cond_image_warped.size(2) and kpts1_all[:,
                                                                 0].min() >= 0, f"kpts1_all[:, 0] max {kpts1_all[:,0]. max()} out of bounds"
    assert kpts1_all[:, 1].max() < cond_image_warped.size(1) and kpts1_all[:,
                                                                 1].min() >= 0, f"kpts1_all[:, 1] max {kpts1_all[:,1]. max()} out of bounds"
    assert kpts2_all[:, 0].max() < cond_image_warped.size(2) and kpts2_all[:,
                                                                 0].min() >= 0, f"kpts2_all[:, 0] max {kpts2_all[:,0]. max()} out of bounds"
    assert kpts2_all[:, 1].max() < cond_image_warped.size(1) and kpts2_all[:,
                                                                 1].min() >= 0, f"kpts2_all[:, 1] max {kpts2_all[:,1]. max()} out of bounds"

    # Use advanced indexing to warp the images
    cond_image_warped[:, kpts2_all[:, 1], kpts2_all[:, 0]] = cond_image_input_warping[:, kpts1_all[:, 1],
                                                             kpts1_all[:, 0]]
    output_image_warped[:, kpts1_all[:, 1], kpts1_all[:, 0]] = output_tensor_input_warping[:, kpts2_all[:, 1],
                                                               kpts2_all[:, 0]]

    cond_image_warped_pil = tensor_to_pil(cond_image_warped)
    output_image_warped_pil = tensor_to_pil(output_image_warped)
    # warped_cond_output_folder = output_folder + f"warpings/warped_{cond_folder_name}/"
    warped_cond_output_folder = os.path.join(output_folder, f"warpings/warped_{cond_folder_name}/")
    create_folder_if_not_exists(warped_cond_output_folder)
    cond_image_warped_pil.save(warped_cond_output_folder + f"{i_opt}.png")
    # wandb.log({f"Cond Image Warped {cond_folder_name}": wandb.Image(cond_image_warped_pil)})

    #warped_target_output_folder = output_folder + f"warpings/warped_{target_folder_name}/"
    warped_target_output_folder = os.path.join(output_folder, f"warpings/warped_{target_folder_name}/")
    create_folder_if_not_exists(warped_target_output_folder)
    output_image_warped_pil.save(warped_target_output_folder + f"{i_opt}.png")
    # wandb.log({f"Warped Image {target_folder_name}": wandb.Image(output_image_warped_pil)})
    time_warping = time.time() - time_pre_warping
    # print("Time Warping: ", time_warping)
    wandb.log({"Warping time": time_warping})
    return cond_image_warped, output_image_warped