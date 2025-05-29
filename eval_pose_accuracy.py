import argparse
import os
import sys
import wandb
from DepthAnything.torchhub.facebookresearch_dinov2_main.dinov2.utils.dtype import as_torch_dtype
from RoMa_functions.DeDoDe.DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from RoMa_functions.DeDoDe.DeDoDe import dedode_detector_L, dedode_descriptor_G
import numpy as np
from RoMa_functions.utils_dedode_pose_accuracy import match_kpts, poselib_pose_accuracy
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import json
from PIL import Image, ImageDraw
import torch
import cv2
from evaluation.recon_metrics import ReconMetricMasked
from evaluation.gen_metrics import GenMetricMasked

import seaborn as sns

def draw_matches(im_A_path, kpts_A, im_B_path, kpts_B,num_show = None):
    if num_show is not None:
        indices = torch.randperm(kpts_A.shape[0])[:num_show]
        kpts_A = kpts_A[indices]
        kpts_B = kpts_B[indices]
    # If kpts are torch
    if isinstance(kpts_A, torch.Tensor):
        kpts_A = kpts_A.cpu().numpy()
        kpts_B = kpts_B.cpu().numpy()
    kpts_A = [cv2.KeyPoint(x,y,1.) for x,y in kpts_A]
    kpts_B = [cv2.KeyPoint(x,y,1.) for x,y in kpts_B]
    matches_A_to_B = [cv2.DMatch(idx, idx, 0.) for idx in range(len(kpts_A))]
    im_A = Image.open(im_A_path)
    im_B = Image.open(im_B_path)
    im_A, im_B = np.array(im_A), np.array(im_B)
    ret = cv2.drawMatches(im_A, kpts_A, im_B, kpts_B,
                    matches_A_to_B, None)
    return ret


def draw_keypoints(im_A_path, kpts_A, im_B_path, kpts_B, num_show=None):
    if num_show is not None:
        indices = torch.randperm(kpts_A.shape[0])[:num_show]
        kpts_A = kpts_A[indices]
        kpts_B = kpts_B[indices]
    # If kpts are torch
    if isinstance(kpts_A, torch.Tensor):
        kpts_A = kpts_A.cpu().numpy()
        kpts_B = kpts_B.cpu().numpy()

    im_A = Image.open(im_A_path)
    im_B = Image.open(im_B_path)
    im_A, im_B = np.array(im_A), np.array(im_B)

    # Create a blank image to draw keypoints
    im_A_with_kpts = im_A.copy()
    im_B_with_kpts = im_B.copy()

    # Generate random colors for keypoints
    colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(len(kpts_A))]

    for (x_A, y_A), (x_B, y_B), color in zip(kpts_A, kpts_B, colors):
        cv2.circle(im_A_with_kpts, (int(x_A), int(y_A)), 2, color, -1)
        cv2.circle(im_B_with_kpts, (int(x_B), int(y_B)), 2, color, -1)

    # Concatenate images horizontally
    ret = np.concatenate((im_A_with_kpts, im_B_with_kpts), axis=1)

    return ret

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
        rgb_colors = [(int(r * 255), int(g*255), int(b * 255)) for r, g, b in rgb_colors]


    return rgb_colors

import random
def set_random_seeds(seed=42):
    print("Random Seed Set: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_matching_image(im_B_path_opt, opt_sequences_folder):
    im_B = Image.open(im_B_path_opt)
    im_B_array = np.array(im_B)

    for image_name in os.listdir(opt_sequences_folder):
        if image_name.endswith(".png"):
            image_path = os.path.join(opt_sequences_folder, image_name)
            current_image = Image.open(image_path)
            current_image_array = np.array(current_image)
            if np.array_equal(im_B_array, current_image_array):
                return os.path.splitext(image_name)[0]
    return None

if __name__ == '__main__':
    random_seed = 0
    set_random_seeds(random_seed)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--exp_path", default='default', type=str, help='Path to experiment folder')
    arg_parser.add_argument("--threshold_matcher", default=0.01, type=float, help='threshold for matching keypoints')
    arg_parser.add_argument("--threshold_ransac", default=0.5, type=float, help='threshold for RANSAC')
    args = arg_parser.parse_args()
    # exp_name = args.exp_name
    exp_path = args.exp_path

    threshold_matcher = args.threshold_matcher
    threshold_ransac = args.threshold_ransac


    delta_azimuth_list = [-25]
    print("delta_azimuth_list: ", delta_azimuth_list)

    inlier_show_threshold = 0
    bTresholdInliers = False


    print("Experiment path: ", exp_path)
    exp_name = exp_path.split("/")[-2]
    print("Experiment name: ", exp_name)
    wandb_expname = f"FINAL-EVAL-" + exp_name

    save_path = exp_path + "eval_poselib/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    base_path = exp_path + "scenes"
    print("Base path: ", base_path)
    folder_path_wandb = "wandb/" # Specify the folder path for wandb

    dedode_inlier_matches_folder = save_path + "dedode_inlier_matches/"
    if not os.path.exists(dedode_inlier_matches_folder):
        os.makedirs(dedode_inlier_matches_folder)


    print("Threshold matcher: ", threshold_matcher)
    run_all_exp = wandb.init(
        # set the wandb project where this run will be logged
        project="MegaScenesSeedSelect",
        # dir=folder_path_wandb + "wandb/",
        # track hyperparameters and run metadata
        name=wandb_expname,
        config={
            "threshold_matcher": threshold_matcher,
            "threshold_ransac": threshold_ransac,
        }
    )

    max_ransac_iterations = 100000
    ransac_opt = {"max_iterations": max_ransac_iterations, "max_epipolar_error": threshold_ransac}
    bundle_opt = {}
    file_path_orig = exp_path + "reconmetrics_orig.txt"
    # Load reconstruction metrics
    if os.path.exists(file_path_orig):
        print("Loading Reconstruction Metrics")
        with open(file_path_orig, 'r') as file:
            reconmetrics_orig = json.load(file)

        reconmetrics_orig_dict = {f'reconmetrics_orig/{key}': value for key, value in reconmetrics_orig.items()}
        wandb.log(reconmetrics_orig_dict)
        print(reconmetrics_orig_dict)

        psnr_orig = reconmetrics_orig['psnr']
        ssim_orig = reconmetrics_orig['ssim']
        lpips_orig = reconmetrics_orig['lpips']
        mask_psnr_orig = reconmetrics_orig['mask_psnr']
        mask_ssim_orig = reconmetrics_orig['mask_ssim']
        mask_lpips_orig = reconmetrics_orig['mask_lpips']
        cnt_orig = reconmetrics_orig['cnt']

        file_path_opt = exp_path + "reconmetrics_opt.txt"
        with open(file_path_opt, 'r') as file:
            reconmetrics_opt = json.load(file)

        reconmetrics_opt_dict = {f'reconmetrics_opt/{key}': value for key, value in reconmetrics_opt.items()}
        wandb.log(reconmetrics_opt_dict)
        print(reconmetrics_opt_dict)

        psnr_opt = reconmetrics_opt['psnr']
        ssim_opt = reconmetrics_opt['ssim']
        lpips_opt = reconmetrics_opt['lpips']
        mask_psnr_opt = reconmetrics_opt['mask_psnr']
        mask_ssim_opt = reconmetrics_opt['mask_ssim']
        mask_lpips_opt = reconmetrics_opt['mask_lpips']
        cnt_opt = reconmetrics_opt['cnt']

        assert cnt_orig == cnt_opt, "Different number pairs for opt and orig"
        improvement_psnr = psnr_opt - psnr_orig
        improvement_ssim = ssim_opt - ssim_orig
        improvement_lpips = lpips_orig - lpips_opt
        improvement_mask_psnr = mask_psnr_opt - mask_psnr_orig
        improvement_mask_ssim = mask_ssim_opt - mask_ssim_orig
        improvement_mask_lpips = mask_lpips_orig - mask_lpips_opt


    # Load generation metrics
    file_path_orig_genmetrics = exp_path + "genmetrics_orig.txt"
    if os.path.exists(file_path_orig_genmetrics):
        print("Loading Gen Metrics")
        with open(file_path_orig_genmetrics, 'r') as file:
            genmetrics_orig = json.load(file)

        genmetrics_orig_dict = {f'genmetrics_orig/{key}': value for key, value in genmetrics_orig.items()}
        wandb.log(genmetrics_orig_dict)
        print(genmetrics_orig_dict)
        fid_orig = genmetrics_orig["fid"]
        masked_fid_orig = genmetrics_orig["masked_fid"]

        file_path_opt_genmetrics = exp_path + "genmetrics_opt.txt"
        with open(file_path_opt_genmetrics, 'r') as file:
            genmetrics_opt = json.load(file)

        genmetrics_opt_dict = {f'genmetrics_opt/{key}': value for key, value in genmetrics_opt.items()}
        wandb.log(genmetrics_opt_dict)
        print(genmetrics_opt_dict)
        fid_opt = genmetrics_opt["fid"]
        masked_fid_opt = genmetrics_opt["masked_fid"]

        improvement_fid = fid_orig - fid_opt
        improvement_masked_fid = masked_fid_orig - masked_fid_opt

        metrics_improvement_dict = {"metrics_improvement/psnr": improvement_psnr,
                                    "metrics_improvement/ssim": improvement_ssim,
                                    "metrics_improvement/lpips": improvement_lpips,
                                    "metrics_improvement/mask_psnr": improvement_mask_psnr,
                                    "metrics_improvement/mask_ssim": improvement_mask_ssim,
                                    "metrics_improvement/mask_lpips": improvement_mask_lpips,
                                    "metrics_improvement/fid": improvement_fid,"metrics_improvement/masked_fid": improvement_masked_fid}
        print("Metrics improvement dict: ", metrics_improvement_dict)
        wandb.log(metrics_improvement_dict)
        print("Metrics added to wandb!")


    bUseConsistencyLoss = False
    if bUseConsistencyLoss:
        original_loss = np.load(exp_path + "original_consistency_loss.npy")
        best_loss = np.load(exp_path + "best_consistency_loss.npy")
        improvement_loss = np.load(exp_path + "improvement_consistency_loss.npy")
        print("len original loss: ", len(original_loss))
        assert np.allclose((best_loss + improvement_loss), original_loss), "Sum of best loss and improvement loss is not equal to original loss"
    else:
        best_loss = np.zeros(100)
        original_loss = np.zeros(100)

    # Define wandb step
    run_all_exp.define_metric("All/step")
    run_all_exp.define_metric("All/*", step_metric="All/step")

    run_all_exp.define_metric("Views/step")
    run_all_exp.define_metric("Views/*", step_metric="Views/step")

    detector = dedode_detector_L(weights=None)
    descriptor = dedode_descriptor_G(weights=None,
                                     dinov2_weights=None)  # You can manually load dinov2 weights, or we'll pull from facebook
    matcher = DualSoftMaxMatcher()

    scene_names = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name))]
    scene_names = sorted(scene_names)
    print("Scene names: ", scene_names)
    num_scenes = len(scene_names)
    print("Number of scenes: ", num_scenes)

    reconmetric_orig = ReconMetricMasked(device='cuda')
    genmetric_orig = GenMetricMasked(device='cuda')

    reconmetric_opt = ReconMetricMasked(device='cuda')
    genmetric_opt = GenMetricMasked(device='cuda')

    colors_values = generate_distinct_colors(num_scenes, rescale=True)
    run_number_all = 0
    view_idxs = list(range(1, len(delta_azimuth_list) + 1))

    view_counter = -1
    all_mean_rot_dist_orig = []
    all_mean_t_angle_dist_orig = []
    all_mean_rot_dist_opt = []
    all_mean_t_angle_dist_opt = []

    all_mask_psnr_orig = []
    all_mask_ssim_orig = []
    all_mask_lpips_orig = []
    all_mask_fid_orig = []
    all_mask_psnr_opt = []
    all_mask_ssim_opt = []
    all_mask_lpips_opt = []
    all_mask_fid_opt = []

    for view_idx in view_idxs:
        all_rot_dists_original = []
        all_t_angle_dists_original = []

        all_rot_dists_optimized = []
        all_t_angle_dists_optimized = []
        max_rot_dist_original = 0
        max_rot_dist_opt = 0

        all_rot_dists_both = np.zeros([len(scene_names), 2])
        # Create two separate figures and axes
        fig_rot, ax_rot = plt.subplots()
        fig_consistency_all, ax_consistency_all = plt.subplots()
        ax_consistency_all.scatter([], [], color='red', label="Original")
        ax_consistency_all.scatter([], [], color='green', label="Optimized")
        # Track added labels
        added_labels = set()
        num_skipped = 0

        current_azimuth = np.abs(delta_azimuth_list[view_idx - 1])
        view_counter += 1
        scene_counter = 0
        print("--------------------------------------View idx: ", view_idx, "--------------------------------------")
        eval_view_folder = os.path.join(save_path, f"eval_view_azimuth_{current_azimuth}")
        if not os.path.exists(eval_view_folder):
            os.makedirs(eval_view_folder)

        max_consistency_loss_all = 0
        max_rot_scene_all = 0
        counter_mask = 0
        for scene_name in scene_names:
            print(f"-------------------{scene_name}-------------------")
            scene_path = f"{base_path}/{scene_name}"
            scene_rot_dist_original = []
            scene_t_angle_dist_original = []
            scene_rot_dist_optimized = []
            scene_t_angle_dist_optimized = []

            # Define wandb step
            run_all_exp.define_metric(f"{scene_name}/step")
            run_all_exp.define_metric(f"{scene_name}/*", step_metric=f"{scene_name}/step")

            fig_consistency, ax_consistency = plt.subplots()
            ax_consistency.scatter([], [], color='red', label="Original")
            ax_consistency.scatter([], [], color='green', label="Optimized")
            max_consistency_loss = 0
            max_rot_scene = 0

            folder_path = scene_path
            im_A_path = folder_path + "/reference.png"

            # for view_idx in view_idxs:
            print(f"----view azimuth {current_azimuth}----")

            im_B_path_orig = folder_path + f"/{current_azimuth}_megascenes.png"
            # skip if im_B_path_orig does not exist
            if not os.path.exists(im_B_path_orig):
                print(f"Original generated image does not exist!")
                continue
            im_B_path_opt = folder_path + f"/{current_azimuth}_opt.png"
            if not os.path.exists(im_B_path_opt):
                print(f"Optimized generated image does not exist!")
                continue
            bComputeMaskedMetrics = True
            if bComputeMaskedMetrics:
                warping_path = folder_path + f"/{current_azimuth}_warping.png"
                warping_image = Image.open(warping_path).convert("RGB")
                warping_array_np = np.array(warping_image)
                # Convert warping array to tensor between -1 and 1
                warping_array = (warping_array_np / 255.) * 2. - 1.
                warping_array = torch.from_numpy(warping_array).cuda().float()

                # Create a mask where warping exists (non-black pixels)
                mask = np.any(warping_array_np != [0, 0, 0], axis=-1)

                # Create a new image with white where warping exists and black otherwise
                warped_mask_array = np.zeros_like(warping_array_np)
                warped_mask_array[mask] = [255, 255, 255]

                # Convert the array back to an image
                warped_mask = Image.fromarray(warped_mask_array)

                # Save the output image
                save_warping_path = eval_view_folder + f"/{current_azimuth}_{counter_mask}_warping.png"
                save_warping_mask_path = eval_view_folder + f"/{current_azimuth}_{counter_mask}_mask.png"

                warping_image.save(save_warping_path)
                warped_mask.save(save_warping_mask_path)

                im_B_orig = Image.open(im_B_path_orig)
                im_B_opt = Image.open(im_B_path_opt)
                im_B_orig_array_masked_np = np.array(im_B_orig)*(warped_mask_array/255.).astype(np.uint8)
                im_B_orig_array_masked = (im_B_orig_array_masked_np/255.)*2. - 1.
                im_B_orig_array_masked = torch.from_numpy(im_B_orig_array_masked)

                im_B_opt_array_masked_np = np.array(im_B_opt)*(warped_mask_array/255.).astype(np.uint8)
                im_B_opt_array_masked = (im_B_opt_array_masked_np/255.)*2. - 1.
                im_B_opt_array_masked = torch.from_numpy(im_B_opt_array_masked)

                im_B_orig_masked = Image.fromarray(im_B_orig_array_masked_np)
                im_B_opt_masked = Image.fromarray(im_B_opt_array_masked_np)
                im_B_orig_masked.save(save_warping_path.replace("_warping.png", "_masked_orig.png"))
                im_B_opt_masked.save(save_warping_path.replace("_warping.png", "_masked_opt.png"))

                _ = reconmetric_orig.update(warping_array.unsqueeze(0), im_B_orig_array_masked.cuda().float().unsqueeze(0))
                _ = genmetric_orig.update(warping_array.unsqueeze(0), im_B_orig_array_masked.cuda().float().unsqueeze(0))

                _ = reconmetric_opt.update(warping_array.unsqueeze(0), im_B_opt_array_masked.cuda().float().unsqueeze(0))
                _ = genmetric_opt.update(warping_array.unsqueeze(0), im_B_opt_array_masked.cuda().float().unsqueeze(0))

            consistency_loss_original_path = folder_path + f"/{current_azimuth}/original_consistency_loss.txt"
            consistency_loss_optimized_path = folder_path + f"/{current_azimuth}/opt_consistency_loss.txt"
            print("Folder Path: ", folder_path)
            K = np.load(folder_path + "/K.npy")
            # Check if im_B_path_opt exists
            if not os.path.exists(im_B_path_opt):
                print(f"Optimized generated image does not exist!")
                continue

            bUseRoMaMatches = False
            if bUseRoMaMatches:
                keypoints_loading_folder = folder_path + f"/{current_azimuth}/analysis/keypoints"
                kpts_A_orig = np.load(keypoints_loading_folder + f"/kpts1_0.npy")
                kpts_B_orig = np.load(keypoints_loading_folder + f"/kpts2_0.npy")

                best_i_opt_text_path = folder_path + f"/{current_azimuth}/best_i_opt.txt"
                if os.path.exists(best_i_opt_text_path):
                    print("Loading best i opt from file")
                    with open(best_i_opt_text_path, 'r') as file:
                        best_i_opt = int(file.read().strip())
                else:
                    print("Using find_matching_image to find best_i_opt")
                    opt_sequences_folder = folder_path + f"/{current_azimuth}/opt_sequence"
                    best_i_opt = find_matching_image(im_B_path_opt, opt_sequences_folder)
                print(f"Best i opt for scene {scene_name} and view azimuth {current_azimuth}: ", best_i_opt)

                kpts_A_opt = np.load(keypoints_loading_folder + f"/kpts1_{best_i_opt}.npy")
                kpts_B_opt = np.load(keypoints_loading_folder + f"/kpts2_{best_i_opt}.npy")
            else:
                print("im_B_path_orig: ", im_B_path_orig)
                kpts_A_orig, kpts_B_orig = match_kpts(im_A_path, im_B_path_orig, detector, descriptor, matcher, threshold=threshold_matcher)
                kpts_A_opt, kpts_B_opt = match_kpts(im_A_path, im_B_path_opt, detector, descriptor, matcher, threshold=threshold_matcher)

            orbitposes = np.load(folder_path+"/orbitposes.npy")
            P_input = orbitposes[0,:,:]
            assert np.allclose(P_input, np.eye(4)), "input_pose is not an identity matrix"
            P_gt = orbitposes[view_idx,:,:]
            P_gt = np.linalg.inv(P_gt)

            P_gen_orig, rot_dist_deg_orig, t_angle_dist_deg_orig, inliers_orig = poselib_pose_accuracy(
                P_gt, kpts_A_orig, kpts_B_orig, K, ransac_opt=ransac_opt, bundle_opt=bundle_opt)
            num_inliers_orig = np.sum(inliers_orig)
            # Skip if num_inlers_orig is less than 70
            if num_inliers_orig < inlier_show_threshold and bTresholdInliers:
                num_skipped += 1
                print(f"Number of inliers less than {inlier_show_threshold}")
                continue

            # if rot_dist_deg_orig<15:
            all_rot_dists_original.append(rot_dist_deg_orig)
            all_t_angle_dists_original.append(t_angle_dist_deg_orig)
            scene_rot_dist_original.append(rot_dist_deg_orig)
            scene_t_angle_dist_original.append(t_angle_dist_deg_orig)

            P_gen_opt, rot_dist_deg_opt, t_angle_dist_deg_opt, inliers_opt = poselib_pose_accuracy(
                P_gt, kpts_A_opt, kpts_B_opt, K, ransac_opt=ransac_opt, bundle_opt=bundle_opt)
            num_inliers_opt = np.sum(inliers_opt)
            bSaveDeDoDe = False
            if bSaveDeDoDe:
                # Filter the keypoints
                kpts_A_orig_filtered = kpts_A_orig[inliers_orig]
                kpts_B_orig_filtered = kpts_B_orig[inliers_orig]

                kpts_A_opt_filtered = kpts_A_opt[inliers_opt]
                kpts_B_opt_filtered = kpts_B_opt[inliers_opt]

                # Show inlier matches
                num_show_orig = kpts_A_orig_filtered.shape[0]
                matches_image_orig = Image.fromarray(
                    draw_matches(im_A_path, kpts_A_orig_filtered, im_B_path_orig, kpts_B_orig_filtered, num_show_orig))
                num_show_opt = kpts_A_opt_filtered.shape[0]
                matches_image_opt = Image.fromarray(
                    draw_matches(im_A_path, kpts_A_opt_filtered, im_B_path_opt, kpts_B_opt_filtered, num_show_opt))

                # Show inlier matches few
                num_show_orig = 30
                matches_image_orig_few = Image.fromarray(
                    draw_matches(im_A_path, kpts_A_orig_filtered, im_B_path_orig, kpts_B_orig_filtered, num_show_orig))
                num_show_opt = kpts_A_opt_filtered.shape[0]
                matches_image_opt_few = Image.fromarray(
                    draw_matches(im_A_path, kpts_A_opt_filtered, im_B_path_opt, kpts_B_opt_filtered, num_show_orig))

                num_show_orig = kpts_A_orig_filtered.shape[0]
                keypoints_image_orig = Image.fromarray(
                    draw_keypoints(im_A_path, kpts_A_orig_filtered, im_B_path_orig, kpts_B_orig_filtered, num_show_orig))
                num_show_opt = kpts_A_opt_filtered.shape[0]
                keypoints_image_opt = Image.fromarray(
                    draw_keypoints(im_A_path, kpts_A_opt_filtered, im_B_path_opt, kpts_B_opt_filtered, num_show_opt))

                # folder_path_dedode_matches = "/mimer/NOBACKUP/groups/snic2022-6-266/josef/MegaScenes/ShowDeDoDeMatches/"
                dedode_inlier_matches_scene_folder = dedode_inlier_matches_folder + f"matches/{scene_name}/"
                dedode_inlier_matches_scene_folder_few = dedode_inlier_matches_folder + f"matches_few/{scene_name}/"

                dedode_inlier_keypoints_scene_folder = dedode_inlier_matches_folder + f"keypoints/{scene_name}/"
                if not os.path.exists(dedode_inlier_matches_scene_folder):
                    os.makedirs(dedode_inlier_matches_scene_folder)
                if not os.path.exists(dedode_inlier_keypoints_scene_folder):
                    os.makedirs(dedode_inlier_keypoints_scene_folder)
                if not os.path.exists(dedode_inlier_matches_scene_folder_few):
                    os.makedirs(dedode_inlier_matches_scene_folder_few)

                save_path_matches_orig = dedode_inlier_matches_scene_folder + f"view_idx_{current_azimuth}_matches_orig.png"
                save_path_matches_opt = dedode_inlier_matches_scene_folder + f"view_idx_{current_azimuth}_matches_opt.png"
                matches_image_orig.save(save_path_matches_orig)
                matches_image_opt.save(save_path_matches_opt)

                save_path_matches_orig_few = dedode_inlier_matches_scene_folder_few + f"view_idx_{current_azimuth}_matches_orig_few.png"
                save_path_matches_opt_few = dedode_inlier_matches_scene_folder_few + f"view_idx_{current_azimuth}_matches_opt_few.png"
                matches_image_orig_few.save(save_path_matches_orig_few)
                matches_image_opt_few.save(save_path_matches_opt_few)

                keypoints_image_orig.save(dedode_inlier_keypoints_scene_folder + f"view_idx_{current_azimuth}_keypoints_orig.png")
                keypoints_image_opt.save(dedode_inlier_keypoints_scene_folder + f"view_idx_{current_azimuth}_keypoints_opt.png")

            # if rot_dist_deg_orig < 15:
            if rot_dist_deg_orig > max_rot_dist_original:
                max_rot_dist_original = rot_dist_deg_orig
            if rot_dist_deg_opt > max_rot_dist_opt:
                max_rot_dist_opt = rot_dist_deg_opt

            scene_color = colors_values[scene_counter]
            # print("Scene color: ", scene_color)
            if scene_name not in added_labels:
                ax_rot.scatter(rot_dist_deg_orig, rot_dist_deg_opt, color=scene_color, label=scene_name)
                added_labels.add(scene_name)
            else:
                ax_rot.scatter(rot_dist_deg_orig, rot_dist_deg_opt, color=scene_color)


            all_rot_dists_both[scene_counter,0] = rot_dist_deg_orig
            all_rot_dists_both[scene_counter,1] = rot_dist_deg_opt

            with open(consistency_loss_original_path, 'r') as file:
                current_loss_original = float(file.read().strip())

            with open(consistency_loss_optimized_path, 'r') as file:
                current_loss_optimized = float(file.read().strip())
            # current_loss_best = best_loss[run_number]
            if current_loss_original > max_consistency_loss:
                max_consistency_loss = current_loss_original
            if current_loss_optimized > max_consistency_loss:
                max_consistency_loss = current_loss_optimized

            if current_loss_original > max_consistency_loss_all:
                max_consistency_loss_all = current_loss_original
            if current_loss_optimized > max_consistency_loss_all:
                max_consistency_loss_all = current_loss_optimized

            if rot_dist_deg_orig > max_rot_scene:
                max_rot_scene = rot_dist_deg_orig
            if rot_dist_deg_opt > max_rot_scene:
                max_rot_scene = rot_dist_deg_opt
            if rot_dist_deg_orig > max_rot_scene_all:
                max_rot_scene_all = rot_dist_deg_orig
            if rot_dist_deg_opt > max_rot_scene_all:
                max_rot_scene_all = rot_dist_deg_opt

            ax_consistency.scatter(current_loss_original, rot_dist_deg_orig, color='red')
            ax_consistency.scatter(current_loss_optimized, rot_dist_deg_opt, color='green')
            ax_consistency.plot([current_loss_original, current_loss_optimized], [rot_dist_deg_orig, rot_dist_deg_opt], 'k--')

            ax_consistency_all.scatter(current_loss_original, rot_dist_deg_orig, color='red')
            ax_consistency_all.scatter(current_loss_optimized, rot_dist_deg_opt, color='green')
            ax_consistency_all.plot([current_loss_original, current_loss_optimized], [rot_dist_deg_orig, rot_dist_deg_opt], 'k--')

            all_rot_dists_optimized.append(rot_dist_deg_opt)
            all_t_angle_dists_optimized.append(t_angle_dist_deg_opt)
            scene_rot_dist_optimized.append(rot_dist_deg_opt)
            scene_t_angle_dist_optimized.append(t_angle_dist_deg_opt)

            print(f"Original Num Matches {kpts_A_orig.shape[0]}, Optimized Num Matches {kpts_A_opt.shape[0]}")
            print(f"Original Num Inliers: {num_inliers_orig}, Optimized Num Inliers: {num_inliers_opt}")
            print(f"Rot angle original: {rot_dist_deg_orig:.3f}, translation angle original {t_angle_dist_deg_orig:.3f}")
            print(f"Rot angle opt: {rot_dist_deg_opt:.3f}, translation angle opt: {t_angle_dist_deg_opt:.3f}")

            run_all_exp.log({
                f"{scene_name}/original_rot_dist": rot_dist_deg_orig,
                f"{scene_name}/original_t_angle_dist": t_angle_dist_deg_orig,
                f"{scene_name}/optimized_rot_dist": rot_dist_deg_opt,
                f"{scene_name}/optimized_t_angle_dist": t_angle_dist_deg_opt,
                f"{scene_name}/improvement_rot_dist": rot_dist_deg_orig - rot_dist_deg_opt,
                f"{scene_name}/improvement_t_angle_dist": t_angle_dist_deg_orig - t_angle_dist_deg_opt,
                f"{scene_name}/Original Num Inliers": num_inliers_orig,
                f"{scene_name}/Optimized Num Inliers": num_inliers_opt,
            })

            run_all_exp.log({
                f"Views//original_rot_dist": rot_dist_deg_orig,
                f"All/original_t_angle_dist": t_angle_dist_deg_orig,
                f"All/optimized_rot_dist": rot_dist_deg_opt,
                f"All/optimized_t_angle_dist": t_angle_dist_deg_opt,
                f"All/improvement_rot_dist": rot_dist_deg_orig - rot_dist_deg_opt,
                f"All/improvement_t_angle_dist": t_angle_dist_deg_orig - t_angle_dist_deg_opt,
                f"All/Original Num Inliers": num_inliers_orig,
                f"All/Optimized Num Inliers": num_inliers_opt,
                "All/step": run_number_all,
            })

            # Log mean and improvement for scene
            wandb.log({
                f"{scene_name}/mean_rot_dist_original": np.mean(scene_rot_dist_original),
                f"{scene_name}/mean_t_angle_dist_original": np.mean(scene_t_angle_dist_original),
                f"{scene_name}/mean_rot_dist_optimized": np.mean(scene_rot_dist_optimized),
                f"{scene_name}/mean_t_angle_dist_optimized": np.mean(scene_t_angle_dist_optimized),
                f"{scene_name}/improvement_rot_dist": np.mean(scene_rot_dist_original) - np.mean(scene_rot_dist_optimized),
                f"{scene_name}/improvement_t_angle_dist": np.mean(scene_t_angle_dist_original) - np.mean(scene_t_angle_dist_optimized),
            })

            print(
                f"Mean rot angle error original: {np.mean(scene_rot_dist_original):.3f}, mean translation angle error original: {np.mean(scene_t_angle_dist_original):.3f}")
            print(
                f"Mean rot angle error optimized: {np.mean(scene_rot_dist_optimized):.3f}, mean translation angle error optimized: {np.mean(scene_t_angle_dist_optimized):.3f}")

            # Create consistency error scatter plot
            ax_consistency.set_xlabel('Consistency Loss')
            ax_consistency.set_ylabel('Rot dist')
            ax_consistency.set_title(f'{scene_name}: Scatter Plot of consistency loss vs rot dist')
            max_axis = max(max_consistency_loss, max_rot_scene)
            # Set x and y axis limits
            ax_consistency.set_xlim(0, max_consistency_loss + 0.5)
            ax_consistency.set_ylim(0, max_rot_scene + 0.5)
            # Add a legend
            ax_consistency.legend()
            # Save the plot to a file
            consistency_plot_filename = eval_view_folder + f"/consistency_error_scatter_plot_{scene_name}.png"
            fig_consistency.savefig(consistency_plot_filename)
            # Log the plot to wandb
            wandb.log({"Consistency Error Scatter Plot": wandb.Image(consistency_plot_filename)})

            scene_counter+=1


        # Create rotation error scatter plot
        ax_rot.set_xlabel('Original rot dist')
        ax_rot.set_ylabel('Optimized rot dist')
        ax_rot.set_title('Scatter Plot of original vs optimized rot dist')
        # Set x and y axis limits
        max_max_value = max(max_rot_dist_original, max_rot_dist_opt)

        extra_border = 0.5
        ax_rot.set_xlim(0, max_max_value + extra_border)
        ax_rot.set_ylim(0, max_max_value + extra_border)
        # Give me max value of x and y axis
        ax_rot.plot([0, max_max_value + extra_border], [0, max_max_value + extra_border], color='red', linestyle='--')
        # Save the plot to a file
        # Give me path one level above base path
        ax_rot.legend()
        rot_plot_filename = eval_view_folder + "/rot_error_scatter_plot.png"
        rot_plot_filename_direct = save_path + f"/rot_error_scatter_plot_{current_azimuth}.png"


        fig_rot.savefig(rot_plot_filename)
        fig_rot.savefig(rot_plot_filename_direct)

        # Create sns kde plot
        all_rot_dists_both_df = pd.DataFrame(all_rot_dists_both, columns=["Original Rot Dist", "Optimized Rot Dist"])

        fig_kde, ax_kde = plt.subplots()


        sns.kdeplot(
            data=all_rot_dists_both_df, x="Original Rot Dist", y="Optimized Rot Dist", fill=True, ax=ax_kde)
        ax_kde.set_xlim(0, max_max_value + extra_border)
        ax_kde.set_ylim(0, max_max_value + extra_border)
        ax_kde.plot([0, max_max_value + extra_border], [0, max_max_value + extra_border], color='red', linestyle='--')

        kde_plot_filename = eval_view_folder + "/rot_error_kde_plot.png"
        fig_kde.savefig(kde_plot_filename)
        kde_plot_filename_direct = save_path + f"/rot_error_kde_plot_{current_azimuth}.png"
        fig_kde.savefig(kde_plot_filename_direct)

        # Create consistency all error scatter plot
        ax_consistency_all.set_xlabel('Consistency Loss')
        ax_consistency_all.set_ylabel('Rot dist')
        ax_consistency_all.set_title('Scatter Plot of consistency loss vs rot dist for all scenes')
        # Set x and y axis limits
        ax_consistency_all.set_xlim(0, max_consistency_loss_all + 0.5)
        ax_consistency_all.set_ylim(0, max_rot_scene_all + 0.5)
        # Add a legend
        ax_consistency_all.legend()
        # Save the plot to a file
        consistency_plot_filename_direct = save_path + f"/consistency_error_scatter_plot_{current_azimuth}.png"
        fig_consistency_all.savefig(consistency_plot_filename_direct)


        # Log the plot to wandb
        wandb.log({"Rotation Error Scatter Plot": wandb.Image(rot_plot_filename)})

        axis_lim_smaller = 5
        # Create a copy of the plot with axis limits set to 20
        fig_rot_copy, ax_rot_copy = plt.subplots()
        ax_rot_copy.set_xlabel('Original rot dist')
        ax_rot_copy.set_ylabel('Optimized rot dist')
        ax_rot_copy.set_title(f'Scatter Plot of original vs optimized rot dist (max {axis_lim_smaller})')
        # Set x and y axis limits to 20
        ax_rot_copy.set_xlim(0, axis_lim_smaller)
        ax_rot_copy.set_ylim(0, axis_lim_smaller)

        # Copy data points from the original plot to the new plot
        for line in ax_rot.get_lines():
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            ax_rot_copy.scatter(x_data, y_data, label=line.get_label(), color=line.get_color())

        # Copy scatter plots (if any)
        for collection in ax_rot.collections:
            ax_rot_copy.scatter(collection.get_offsets()[:, 0], collection.get_offsets()[:, 1],
                                color=collection.get_facecolor(), label=collection.get_label())

        ax_rot_copy.plot([0, axis_lim_smaller], [0, axis_lim_smaller], color='red', linestyle='--')
        # ax_rot_copy.legend()
        rot_plot_filename_copy = eval_view_folder + f"/rot_error_scatter_plot_max{axis_lim_smaller}.png"
        fig_rot_copy.savefig(rot_plot_filename_copy)
        # Log the plot to wandb
        wandb.log({f"Rotation Error Scatter Plot (max {axis_lim_smaller})": wandb.Image(rot_plot_filename_copy)})

        # Log mean and improvement for all scenes
        reconresult_orig = reconmetric_orig.compute()
        reconmetric_orig.reset()
        genresult_orig = genmetric_orig.compute()
        genresult_orig['masked_fid'] = genresult_orig['masked_fid'].cpu().item()
        genmetric_orig.reset()

        with open(os.path.join(eval_view_folder, 'reconmetrics_orig.txt'), 'w') as file:
            json.dump(reconresult_orig, file, indent=4)
        with open(os.path.join(eval_view_folder, 'genmetrics_orig.txt'), 'w') as file:
            json.dump(genresult_orig, file, indent=4)

        reconresult_opt = reconmetric_opt.compute()
        reconmetric_opt.reset()
        genresult_opt = genmetric_opt.compute()
        genresult_opt['masked_fid'] = genresult_opt['masked_fid'].cpu().item()
        genmetric_opt.reset()

        with open(os.path.join(eval_view_folder, 'reconmetrics_opt.txt'), 'w') as file:
            json.dump(reconresult_opt, file, indent=4)
        with open(os.path.join(eval_view_folder, 'genmetrics_opt.txt'), 'w') as file:
            json.dump(genresult_opt, file, indent=4)


        wandb.log({
            f"Views/mean_rot_dist_original": np.mean(all_rot_dists_original),
            f"Views/mean_t_angle_dist_original": np.mean(all_t_angle_dists_original),
            f"Views/mean_rot_dist_optimized": np.mean(all_rot_dists_optimized),
            f"Views/mean_t_angle_dist_optimized": np.mean(all_t_angle_dists_optimized),
            f"Views/mean improvement_rot_dist": np.mean(all_rot_dists_original) - np.mean(all_rot_dists_optimized),
            f"Views/mean improvement_t_angle_dist": np.mean(all_t_angle_dists_original) - np.mean(all_t_angle_dists_optimized),
            f"Views/median_rot_dist_original": np.median(all_rot_dists_original),
            f"Views/median_t_angle_dist_original": np.median(all_t_angle_dists_original),
            f"Views/median_rot_dist_optimized": np.median(all_rot_dists_optimized),
            f"Views/median_t_angle_dist_optimized": np.median(all_t_angle_dists_optimized),
            f"Views/median improvement_rot_dist": np.median(all_rot_dists_original) - np.median(all_rot_dists_optimized),
            f"Views/median improvement_t_angle_dist": np.median(all_t_angle_dists_original) - np.median(all_t_angle_dists_optimized),
            f"Views/Mask PSNR orig": reconresult_orig['mask_psnr'],
            f"Views/Mask SSIM orig": reconresult_orig['mask_ssim'],
            f"Views/Mask LPIPS orig": reconresult_orig['mask_lpips'],
            f"Views/Mask PSNR opt": reconresult_opt['mask_psnr'],
            f"Views/Mask SSIM opt": reconresult_opt['mask_ssim'],
            f"Views/Mask LPIPS opt": reconresult_opt['mask_lpips'],
            f"Views/Mask PSNR improvement": reconresult_opt['mask_psnr'] - reconresult_orig['mask_psnr'],
            f"Views/Mask SSIM improvement": reconresult_opt['mask_ssim'] - reconresult_orig['mask_ssim'],
            f"Views/Mask LPIPS improvement": reconresult_orig['mask_lpips'] - reconresult_opt['mask_lpips'],
            f"Views/Mask FID orig": genresult_orig['masked_fid'],
            # f"Views/Masked KID orig": genresult_orig['masked_kid'],
            f"Views/Mask FID opt": genresult_opt['masked_fid'],
            # f"Views/Masked KID opt": genresult_opt['masked_kid'],
            f"Views/Mask FID improvement": genresult_orig['masked_fid'] - genresult_opt['masked_fid'],
            # f"Views/Masked KID improvement": genresult_orig['masked_kid'][0] - genresult_opt['masked_kid'][0],
            f"Views/number skipped": num_skipped,
            f"Views/step": current_azimuth,
        })

        print(f"-------------------Summary Azimuth {current_azimuth} -------------------")
        all_mask_psnr_orig.append(reconresult_orig['mask_psnr'])
        all_mask_ssim_orig.append(reconresult_orig['mask_ssim'])
        all_mask_lpips_orig.append(reconresult_orig['mask_lpips'])
        all_mask_fid_orig.append(genresult_orig['masked_fid'])
        all_mask_psnr_opt.append(reconresult_opt['mask_psnr'])
        all_mask_ssim_opt.append(reconresult_opt['mask_ssim'])
        all_mask_lpips_opt.append(reconresult_opt['mask_lpips'])
        all_mask_fid_opt.append(genresult_opt['masked_fid'])


        mean_rot_dist_orig = np.mean(all_rot_dists_original)
        mean_t_angle_dist_orig = np.mean(all_t_angle_dists_original)
        mean_rot_dist_opt = np.mean(all_rot_dists_optimized)
        mean_t_angle_dist_opt = np.mean(all_t_angle_dists_optimized)
        all_mean_rot_dist_orig.append(mean_rot_dist_orig)
        all_mean_t_angle_dist_orig.append(mean_t_angle_dist_orig)
        all_mean_rot_dist_opt.append(mean_rot_dist_opt)
        all_mean_t_angle_dist_opt.append(mean_t_angle_dist_opt)

        print(
            f"Mean rot angle error original: {np.mean(all_rot_dists_original):.3f}, mean translation angle error original: {np.mean(all_t_angle_dists_original):.3f}")
        print(
            f"Mean rot angle error optimized: {np.mean(all_rot_dists_optimized):.3f}, mean translation angle error optimized: {np.mean(all_t_angle_dists_optimized):.3f}")
        print(f"Improvement in mean rot angle error: {np.mean(all_rot_dists_original) - np.mean(all_rot_dists_optimized):.3f}")
        print(
            f"Improvement in mean translation angle error: {np.mean(all_t_angle_dists_original) - np.mean(all_t_angle_dists_optimized):.3f}")

        print(f"Median rot angle error original: {np.median(all_rot_dists_original):.3f}, median translation angle error original: {np.median(all_t_angle_dists_original):.3f}")
        print(
            f"Median rot angle error optimized: {np.median(all_rot_dists_optimized):.3f}, median translation angle error optimized: {np.median(all_t_angle_dists_optimized):.3f}")
        print(f"Improvement in median rot angle error: {np.median(all_rot_dists_original) - np.median(all_rot_dists_optimized):.3f}")
        print(
            f"Improvement in median translation angle error: {np.median(all_t_angle_dists_original) - np.median(all_t_angle_dists_optimized):.3f}")

        print("Total number runs: ", len(all_rot_dists_original))

    print("-------------------Summary All Angles-------------------")


    print(
        f"Mean rot angle error original: {np.mean(all_mean_rot_dist_orig):.3f}, mean translation angle error original: {np.mean(all_mean_t_angle_dist_orig):.3f}")
    print(
        f"Mean rot angle error optimized: {np.mean(all_mean_rot_dist_opt):.3f}, mean translation angle error optimized: {np.mean(all_mean_t_angle_dist_opt):.3f}")
    print(f"Improvement in mean rot angle error: {np.mean(all_mean_rot_dist_orig) - np.mean(all_mean_rot_dist_opt):.3f}")
    print(
        f"Improvement in mean translation angle error: {np.mean(all_mean_t_angle_dist_orig) - np.mean(all_mean_t_angle_dist_opt):.3f}")

    wandb.log({
        f"Final/mean_rot_dist_original": np.mean(all_mean_rot_dist_orig),
        f"Final/mean_t_angle_dist_original": np.mean(all_mean_t_angle_dist_orig),
        f"Final/mean_rot_dist_optimized": np.mean(all_mean_rot_dist_opt),
        f"Final/mean_t_angle_dist_optimized": np.mean(all_mean_t_angle_dist_opt),
        f"Final/mean improvement_rot_dist": np.mean(all_mean_rot_dist_orig) - np.mean(all_mean_rot_dist_opt),
        f"Final/mean improvement_t_angle_dist": np.mean(all_mean_t_angle_dist_orig) - np.mean(all_mean_t_angle_dist_opt),
        f"Final/mask PSNR orig": np.mean(all_mask_psnr_orig),
        f"Final/mask SSIM orig": np.mean(all_mask_ssim_orig),
        f"Final/mask LPIPS orig": np.mean(all_mask_lpips_orig),
        f"Final/mask FID orig": np.mean(all_mask_fid_orig),
        f"Final/mask PSNR opt": np.mean(all_mask_psnr_opt),
        f"Final/mask SSIM opt": np.mean(all_mask_ssim_opt),
        f"Final/mask LPIPS opt": np.mean(all_mask_lpips_opt),
        f"Final/mask FID opt": np.mean(all_mask_fid_opt),
        f"Final/mask PSNR improvement": np.mean(all_mask_psnr_opt) - np.mean(all_mask_psnr_orig),
        f"Final/mask SSIM improvement": np.mean(all_mask_ssim_opt) - np.mean(all_mask_ssim_orig),
        f"Final/mask LPIPS improvement": np.mean(all_mask_lpips_opt) - np.mean(all_mask_lpips_orig),
        f"Final/mask FID improvement": np.mean(all_mask_fid_opt) - np.mean(all_mask_fid_orig),
    })




