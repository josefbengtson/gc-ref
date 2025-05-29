import math
import numpy as np
import time
import torch, torchvision
import torch.nn.functional as F
import torch.nn as nn
torch.cuda.synchronize()

from omegaconf import OmegaConf
from PIL import Image
import cv2
from torchvision import transforms
import os, json, sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import matplotlib.pyplot as plt

import glob, ipdb
from tqdm import tqdm
import yaml
import importlib
from datetime import datetime
import pickle
import hashlib
from os.path import join
import warnings


from dataloader.data_helpers import *
from dataloader.depth_helpers import *
from dataloader.util_3dphoto import unproject_depth, render_view, render_multiviews 

from accelerate import Accelerator
from torch.utils.data import DataLoader
from ldm.util import instantiate_from_config
from ldm.logger import ImageLogger
from accelerate.utils import set_seed

from dataloader.evalhelpers import *

from RoMa_functions.utils_spherical import compute_F
from RoMa_functions.SED import consistency_loss
from RoMa_functions.utils_RoMa import roma_matching, downsample_good_samples, draw_matches, generate_distinct_colors, warp_images
from RoMa_functions.RoMa.roma import roma_outdoor
from RoMa_functions.utils_spherical_3dof import  gen_T, rel_T
import wandb
import sys
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_folder_one_step_up(folderA, new_folder_name):
    # Get the parent directory of folderA
    parent_dir = os.path.dirname(folderA)
    # Create the new folder path
    folderB = os.path.join(parent_dir, new_folder_name)
    # Create the new folder if it does not exist
    os.makedirs(folderB, exist_ok=True)
    return folderB

def create_folder_two_steps_up(folderA, new_folder_name):
    # Get the grandparent directory of folderA
    grandparent_dir = os.path.dirname(os.path.dirname(folderA))
    # Create the new folder path
    folderB = os.path.join(grandparent_dir, new_folder_name)
    # Create the new folder if it does not exist
    os.makedirs(folderB, exist_ok=True)
    return folderB


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def setup_model(args):
    expdir = args.exp_dir
    print("config load path: ", join(expdir, 'config.yaml'))
    config_file = yaml.safe_load(open( join(expdir, 'config.yaml') ))
    train_configs = config_file.get('training', {})
    img_logger = ImageLogger(log_directory=expdir, log_images_kwargs=train_configs['log_images_kwargs'])
    model = instantiate_from_config(config_file['model']).eval()
    accelerator = Accelerator()

    resume_folder = 'latest' if args.resume == -1 else f'iter_{args.resume}'
    print("resume_folder:", resume_folder)
    args.resume = int(open(os.path.join(args.exp_dir, 'latest/iteration.txt'), "r").read()) if args.resume == -1 else args.resume
    print("args.resume: ", args.resume)
    print("loading from iteration {}".format(args.resume))

    if args.ckpt_file:
        print("in load zeronvs.ckpt")
        old_state = torch.load(join(args.exp_dir, resume_folder, 'zeronvs.ckpt'), map_location="cpu")["state_dict"]
        model.load_state_dict(old_state)

    model = accelerator.prepare( model )
    
    if not args.ckpt_file:
        print("in load other")
        print("loading from: ", join(args.exp_dir, resume_folder))
        accelerator.load_state(join(args.exp_dir, resume_folder))
    return model, img_logger


def setup_paths(args, bUseSpiral):
    savepath = args.savepath
    os.makedirs(savepath, exist_ok=True)
    orbitpath = join(savepath, 'orbit')
    os.makedirs(orbitpath, exist_ok=True)

    if bUseSpiral:
        spiralpath = join(savepath, 'spiral')
        os.makedirs(spiralpath, exist_ok=True)


def setup_scene_and_poses(args, num_poses=2, x_end=0.5, rotation_angle=15):
    scene_name = args.scene_name
    split_type = args.split_type

    inputimg = Image.open(args.inputimg)
    assert inputimg.size[0] == 256 and inputimg.size[1] == 256, "Input image is not 256x256"

    refimg = resize_with_padding(inputimg, target_size=256, returnpil=True)
    refimg_nopad = resize_with_padding(inputimg, target_size=256, return_unpadded=True, returnpil=True)

    refimg_nopad.save(join(args.savepath, 'reference.png'))
    refimg_nopad = np.array(refimg_nopad)
    refimg = np.array(refimg)/ 255. # 0-1 for unproject_depth input

    # get depth
    inputimg = np.array(inputimg)/ 255. # use original resolution for depth estimation, but resize depth to refimg shape
    depth_model, dtransform = load_depth_model()
    h, w = refimg_nopad.shape[:2]
    print("h: ", h,", w: ", w)
    img = dtransform({'image': inputimg})['image']
    img = torch.from_numpy(img).unsqueeze(0).cuda()
    with torch.no_grad():
        est_disparity = depth_model(img).cpu()
        est_disparity = F.interpolate(est_disparity[None], (h, w), mode='bicubic', align_corners=False)[0, 0] # bicubic
    depthmap = invert_depth(est_disparity)
  
    


    # Load megascenes intrinsics
    intrinsics_path = args.intrinsics_path
    intrinsics_loaded= np.load(intrinsics_path)
    intrinsics_keys = list(intrinsics_loaded.keys())
    assert len(intrinsics_keys)==1, "Something wrong with intrinsics.npz file"
    K_loaded = intrinsics_loaded[intrinsics_keys[0]]

    focal_length_loaded = K_loaded[0, 0]
    w_loaded = K_loaded[0, 2] * 2
    w_new = 256
    focal_length_new = focal_length_loaded * (w_new / w_loaded)
    print(f"---------- Loading Focal Length Check for scene {scene_name}----------")
    print("Focal length: ", focal_length_loaded)
    print("w_loaded: ", w_loaded)
    print("w_new: ", w_new)
    print("Focan length new: ", focal_length_new)
    fx = focal_length_new
    fy = focal_length_new


    sensor_diagonal = math.sqrt(w**2 + h**2)
    fov = 2 * math.atan(sensor_diagonal / (2 * fx))
    fov = torch.tensor(fov)

    ext1 = np.eye(4)

    # ext2 = ext1.copy()
    K_target = np.eye(3)
    K_target[0, 0], K_target[1, 1] = fx, fy
    K_target[0, 2], K_target[1, 2] = w/2, h/2 # cx,cy is w/2,h/2, respectively
    depthmap = depthmap.numpy()
    scales = np.quantile( depthmap.reshape(-1), q=0.2)
    mean_depth = np.mean(depthmap)
    print("Mean depth: ", mean_depth)
    print("Median depth: ", np.median(depthmap))
    print("20% depth percentile: ", scales)

    depthmap = depthmap / scales
    depthmap = depthmap.clip(0,100)

    K_input = np.eye(3)
    K_input[0, 0], K_input[1, 1] = fx, fy
    K_input[0, 2], K_input[1, 2] = w/2, h/2 # cx,cy is w/2,h/2, respectively

    # get mesh
    plypath = join(args.savepath, 'mesh.ply')
    mesh = unproject_depth(plypath, refimg_nopad/255., depthmap, K_target, np.linalg.inv(ext1), scale_factor=1.0, add_faces=True, prune_edge_faces=True) # takes C2W

    # save warps
    def savewarps(warps):
        warppath = join(args.savepath, 'xtrans/warped')
        os.makedirs(warppath, exist_ok=True)
        warpedimgs = [(w*255).astype(np.uint8) for w in warps]
        pilframes = [Image.fromarray(f) for f in warpedimgs]
        pilframes[0].save(join(warppath,f'warps.gif'), save_all=True, append_images=pilframes[1:], loop=0, duration=100)


    # setup poses 
    # orbit poses
    print(f"Start Pose Parameters - Num Poses: {num_poses}, x_end: {x_end}, rotation_angle: {rotation_angle}")
    bUseSphericalPoses = True
    if bUseSphericalPoses:
        orbitposes = []
        up_vector = (0,0,1)
        radius = 1
        orbitpose_spherical_0 = gen_T(elevation=90, azimuth=0, r=radius, up=up_vector)
        orbitpose_spherical_0_rel = rel_T(orbitpose_spherical_0,orbitpose_spherical_0)
        orbitposes.append(orbitpose_spherical_0_rel.numpy())


        print("args.az_start: ", args.az_start)
        print("args.az_end: ", args.az_end)
        print("args.az_step: ", args.az_step)
        delta_azimuth_list = np.arange(-args.az_start, -args.az_end-0.1, -args.az_step).astype(int).tolist()
        print("delta_azimuth_list: ", delta_azimuth_list)
        # root_folder = os.path.abspath(os.path.join(args.savepath, "../.."))
        delta_azimuth_list_folder = os.path.join(args.savepath, "delta_azimuth_list.txt")
        with open(delta_azimuth_list_folder, 'w') as file:
            for item in delta_azimuth_list:
                file.write(f"{item}\n")


        print("delta_azimuth_list: ", delta_azimuth_list)
        for delta_azimuth in delta_azimuth_list:
            orbitpose_spherical = gen_T(elevation=90, azimuth=0+delta_azimuth, r=radius, up=up_vector)
            orbitpose_spherical_rel = rel_T(orbitpose_spherical, orbitpose_spherical_0)
            orbitposes.append(orbitpose_spherical_rel.numpy())

    else:
        orbitposes = get_orbit_poses_original(num_poses=num_poses, x_end=x_end, rotation_angle=rotation_angle)
    orbitwarps, _  = render_multiviews(h, w, K_target, orbitposes, mesh)

    # Save orbitposes to numpy files
    orbitposes_np = np.array(orbitposes)
    print("orbitposes_np shape: ", orbitposes_np.shape)
    # np.save(join(args.savepath, 'orbitposes.npy'), orbitposes_np)
    np.save(join(args.savepath, 'orbitposes.npy'), orbitposes_np)

    orbit_rel_poses = []
    F_orbits = []
    for ext2 in orbitposes: # change this!!!
        print("-------------------------------------------------------")
        refpose = np.linalg.inv(ext1) # convert to c2w
        rel_pose = np.linalg.inv(refpose) @ ext2 # 4x4
        T1 = torch.from_numpy(ext1).double().to(DEVICE)
        T2 = torch.from_numpy(ext2).double().to(DEVICE)
        T2 = torch.linalg.inv(T2)

        K_input_torch = torch.from_numpy(K_input).double().to(DEVICE)
        K_target_torch = torch.from_numpy(K_target).double().to(DEVICE)

        F_orbit = compute_F(K_input_torch, K_target_torch, T1, T2)
        # print("F orbit: ", F_orbit)
        fov_enc = torch.stack( [fov, torch.sin(fov), torch.cos(fov)] )
        rel_pose = torch.tensor(rel_pose.reshape((16)))
        rel_pose = torch.cat([rel_pose, fov_enc]).float()
        orbit_rel_poses.append(rel_pose)
        F_orbits.append(F_orbit)
        print("-------------------------------------------------------")


        # spiral poses

    spiralposes = get_front_facing_trans(num_frames=20, max_trans=3, z_div=2)
    spiralwarps, _  = render_multiviews(h, w, K_target, spiralposes, mesh)
    
    spiral_rel_poses = []
    for ext2 in spiralposes: # change this!!!
        refpose = np.linalg.inv(ext1) # convert to c2w
        rel_pose = np.linalg.inv(refpose) @ ext2 # 4x4
        
        fov_enc = torch.stack( [fov, torch.sin(fov), torch.cos(fov)] )
        rel_pose = torch.tensor(rel_pose.reshape((16)))
        rel_pose = torch.cat([rel_pose, fov_enc]).float()
        spiral_rel_poses.append(rel_pose)

    return refimg, refimg_nopad, orbitwarps, orbit_rel_poses, spiralwarps, spiral_rel_poses, K_input, K_target, F_orbits, fx, delta_azimuth_list

#




def run_opt(args):
    print("Args.repeat: ", args.repeat)
    wandb_folder_path = "/mimer/NOBACKUP/groups/snic2022-6-266/josef/MegaScenes/"
    experiment_description = args.exp_name
    seed = args.seed
    input_image_idx = args.input_image_idx
    print("Num ddim steps: ", args.ddim_steps)
    scene_name = args.scene_name
    wandb_experiment_description = scene_name + "_" + experiment_description
    certainty_threshold = args.certainty_threshold

    run_current_exp = wandb.init(
        # set the wandb project where this run will be logged
        project="MegaScenesSeedSelect",
        dir=wandb_folder_path + "wandb/",
        # track hyperparameters and run metadata
        name=wandb_experiment_description,
        config={
            "repeat": args.repeat,
            "batch_size": args.batch_size,
            "num_poses": args.num_poses,
            "x_end": args.x_end,
            "rotation_angle": args.rotation_angle,
            "lr": args.lr,
            "alpha_rgb": args.alpha_rgb,
            "seed": seed,
            "num_opt_steps": args.num_opt_steps,
            "input_image_idx": input_image_idx,
            "num_matches": args.num_matches,
            "ddim_steps": args.ddim_steps,
            "scene_name": scene_name,
            "certainty_threshold": certainty_threshold
        }
    )
    print("args dictionary: ", args)
    # define our custom x axis metric

    if not args.debug:
        model, img_logger = setup_model(args)

    bUseSpiral = False


    # Load poses and images
    setup_paths(args, bUseSpiral)
    num_poses = args.num_poses
    x_end = args.x_end
    rotation_angle = args.rotation_angle
    num_opt_steps = args.num_opt_steps
    refimg, refimg_nopad, orbitwarps, orbit_rel_poses, spiralwarps, spiral_rel_poses, K1, K2, F_orbits, f, delta_azimuth_list = setup_scene_and_poses(args, num_poses = num_poses + 1, x_end=x_end, rotation_angle=rotation_angle)
    print("K1: ", K1)
    print("K2: ", K2)
    # Save K1 to K.npy
    np.save(join(args.savepath, 'K.npy'), K1)

    # Load RoMa Matching Model
    print("orbitwarps shape: ", orbitwarps.shape)
    roma_model = roma_outdoor(device=DEVICE)
    roma_model.decoder.detach = False
    roma_model.upsample_preds = False


    h,w = refimg_nopad.shape[:2]
    shortside = min(h,w)
    diff = math.ceil((256-shortside)/2) # round up to avoid padding
    end = 256-diff

    # Convert Refimg to tensor and normalize
    batchsize = args.batch_size
    refimg = resize_with_padding(Image.fromarray((refimg*255).astype(np.uint8)), target_size=256)/127.5-1
    refimg_path = join(args.savepath, 'reference.png')
    run_current_exp.log({"Reference Image": [wandb.Image(refimg_path)]})

    refimg_tensor = torch.from_numpy(refimg)
    print("Ref image investigation")
    print("refimg_tensor step 1")
    print("refimg_tensor shape: ", refimg_tensor.shape)
    print("refimg_tensor min: ", torch.min(refimg_tensor))
    print("refimg_tensor max: ", torch.max(refimg_tensor))
    refimg_tensor = (refimg_tensor.permute(2, 0, 1)).float()
    # refimg_tensor = ((refimg_tensor.permute(2, 0, 1) + 1) * 127.5).float()

    print("refimg_tensor step 2")
    print("refimg_tensor shape: ", refimg_tensor.shape)
    print("refimg_tensor min: ", torch.min(refimg_tensor))
    print("refimg_tensor max: ", torch.max(refimg_tensor))

    outputs = []

    cfg_scale = args.cfg

    # Create Folders
    print("Length of orbit poses: ", len(orbit_rel_poses))

    original_sequence_folder = join(args.savepath, f'original_sequence/')
    create_folder_if_not_exists(original_sequence_folder)

    both_sequence_folder = join(args.savepath, f'both_sequence/')
    create_folder_if_not_exists(both_sequence_folder)

    optimized_sequence_folder = join(args.savepath, f'optimized_sequence/')
    create_folder_if_not_exists(optimized_sequence_folder)

    warped_save_folder = join(args.savepath, f'warped_images/')
    create_folder_if_not_exists(warped_save_folder)

    all_sequence_folder = create_folder_two_steps_up(args.savepath, "all_sequences/")

    # Set parameters for matching and optimization
    num_matches = args.num_matches
    bCertaintySampling = True

    alpha_rgb = args.alpha_rgb
    lr = args.lr
    good_samples = None
    ddim_steps = args.ddim_steps

    inlier_threshold_huber_loss = 2
    huber_loss = nn.HuberLoss(delta=inlier_threshold_huber_loss)
    l1_loss = nn.L1Loss()

    # Orbit poses
    num_orbit_poses = len(orbit_rel_poses)
    i_list = range(1, num_orbit_poses)

    print("i_list: ", i_list)

    newdataloader = {}
    bUse_ug = False

    print("Initializing manual random seed: ", seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    torch.backends.cudnn.benchmark = True

    for i in i_list:
        current_azimuth = np.abs(delta_azimuth_list[i-1])
        current_View = f"View_{current_azimuth}"
        run_current_exp.define_metric(current_View + "/step")
        run_current_exp.define_metric(current_View + f"/*", step_metric=current_View + "/step")


        print("args.savepath: ", args.savepath)
        # Create folders
        analysis_folder = join(args.savepath, f'{current_azimuth}',"analysis")
        print("Analysis folder: ",analysis_folder)
        create_folder_if_not_exists(analysis_folder)

        matches_folder = join(analysis_folder, 'matches')
        create_folder_if_not_exists(matches_folder)

        keypoints_folder = join(analysis_folder, 'keypoints')
        create_folder_if_not_exists(keypoints_folder)

        epipolar_both_no_matches_folder = join(analysis_folder, 'epipolar_both_no_matches')
        create_folder_if_not_exists(epipolar_both_no_matches_folder)

        histogram_folder = join(analysis_folder, 'histograms')
        create_folder_if_not_exists(histogram_folder)


        opt_sequence_folder = join(args.savepath, f'{current_azimuth}','opt_sequence/')
        create_folder_if_not_exists(opt_sequence_folder)


        print("Orbit pose: ", i)

        warped_img_save_path = join(warped_save_folder, f"{current_azimuth}.png")
        warped_img_save_path_direct = join(args.savepath, f"{current_azimuth}_warping.png")
        warpedimg = (orbitwarps[i]*255).astype(np.uint8)
        Image.fromarray(warpedimg).save(warped_img_save_path)
        Image.fromarray(warpedimg).save(warped_img_save_path_direct)
        lwarp = resize_with_padding(Image.fromarray(warpedimg), target_size=32)/127.5-1
        highwarp = torch.tensor(resize_with_padding(Image.fromarray(warpedimg), target_size=256)/127.5-1).permute(2,0,1).unsqueeze(0).repeat(batchsize,1,1,1)

        # Load relative pose and Fundamental matrix F
        rel_pose = orbit_rel_poses[i].unsqueeze(0).repeat(batchsize,1)

        F_i = F_orbits[i]
        F_i = F_i.to(DEVICE)

        dataloader = dict(image_target=[np.zeros((256,256,3))-1.0], image_ref=[refimg], warped_depth=[lwarp], rel_pose=rel_pose) # txt=[""*batchsize], 
        if args.zeronvs:
            del dataloader['warped_depth']
        if args.warponly:
            del dataloader['rel_pose']

        for k in dataloader.keys():
            if k not in ['rel_pose'] :
                print("batchsize in dataloader repeat loop:", batchsize)
                dataloader[k] = torch.tensor(dataloader[k][0]).float().unsqueeze(0).repeat(batchsize,1,1,1)

        for k in dataloader.keys():
            if k in newdataloader:
                newdataloader[k] = torch.cat([newdataloader[k], dataloader[k]])
            else:
                newdataloader[k] = dataloader[k]

        if (i%args.repeat==0 and i!=0) or i==len(orbit_rel_poses)-1:

            # Initialize optimization for this view
            # Sample initial random latent z_T
            latent_shape = (1, 4, 32, 32)
            z_T_noise = torch.randn(latent_shape, device=DEVICE)
            z_T_start = z_T_noise.clone()
            z_T = nn.Parameter(z_T_noise, requires_grad=True)
            run_current_exp.log({"Change in z_T": torch.norm(z_T - z_T_start)})
            optimizer = torch.optim.Adam([z_T], lr=lr)

            # Optimize for num_opt_steps
            for i_opt in range(num_opt_steps):
                print("i_opt: ", i_opt)

                optimizer.zero_grad()
                print(f"z_T norm at i_opt={i_opt}: ", torch.norm(z_T))
                input_dict = {'good_samples': good_samples, 'certainty_threshold': certainty_threshold,
                                                   'num_matches': num_matches, 'roma_model': roma_model,
                                                  'huber_loss': huber_loss, 'l1_loss': l1_loss, 'h': h, 'w': w,
                                                   'bCertaintySampling': bCertaintySampling,'alpha_rgb': alpha_rgb,
                                                   'output_folder': args.savepath, 'i_pose': i, 'cfg_scale': cfg_scale,
                                                    'bUse_ug': bUse_ug, 'lr': lr, "F_i": F_i,
                                                    'wandb': run_current_exp, 'z_T': z_T, "i_opt": i_opt,
                                                    "ddim_steps": ddim_steps, "current_azimuth": current_azimuth}


                # input_dict = {'bUse_ug': bUse_ug}
                # Generate new image based on the current latent z_T
                out = img_logger.log_img(model, newdataloader, args.resume, input_dict, split='test', foldername='360', returngrid='train', warpeddepth=highwarp, has_target=False, onlyretimg=True)

                npout_tensor = ((out.permute(0,2,3,1)))
                out = torch.clamp(out, -1, 1)
                npout = ((out.permute(0,2,3,1).cpu().detach().numpy()+1)*127.5).astype(np.uint8)
                print("npout shape: ", npout.shape)
                print("npout_tensor shape: ", npout_tensor.shape)


                print("npout -> output_img")
                if w<h:
                    output_img = npout[:,:,diff:end,...] # if h larger, then padding was added to width, so crop width
                    output_img_tensor = npout_tensor[:,:,diff:end,...]
                else:
                    output_img = npout[:,diff:end,...]
                    output_img_tensor = npout_tensor[:,diff:end,...]


                opt_image_path = opt_sequence_folder + f"{i_opt}.png"
                Image.fromarray(output_img[0]).save(opt_image_path)
                if i_opt == 0:
                    gen_sequence_path_direct = os.path.join(args.savepath, f"{current_azimuth}_megascenes.png")
                    original_sequence_path = original_sequence_folder + f"{current_azimuth}.png"
                    original_sequence_ref_path = original_sequence_folder + f"0_ref.png"
                    original_output_img = output_img[0]
                    Image.fromarray(original_output_img).save(gen_sequence_path_direct)
                    Image.fromarray(original_output_img).save(original_sequence_path)
                    Image.fromarray(refimg_nopad).save(original_sequence_ref_path)
                    run_current_exp.log({f"Megascenes Image Azimuth {current_azimuth}": [wandb.Image(gen_sequence_path_direct)]})

                output_img_tensor = output_img_tensor.squeeze(0).permute(2,0,1)   # Squeeze to remove batch dimension
                outputs.append(output_img)
                print("Output shape: ", output_img.shape)

                bEvaluateWarping = False
                if bEvaluateWarping:
                    # warpedimg_tensor = torch.tensor(orbitwarps[i]).reshape(3, 256, 256)
                    warpedimg_tensor = torch.from_numpy(orbitwarps[i])
                    warpedimg_tensor = ((warpedimg_tensor.permute(2, 0, 1) + 1) * 127.5).float()

                    output_img_tensor = warpedimg_tensor/255.

                print("Output image tensor shape: ", output_img_tensor.shape)



                refimg_tensor_rescaled = (refimg_tensor+1)/2 # Rescale to [0, 1]
                output_img_tensor_rescaled = (output_img_tensor+1)/2 # Rescale to [0, 1]
                print("Pre consistency loss check: ")
                print(f"refimg_tensor min: {torch.min(refimg_tensor)}, max: {torch.max(refimg_tensor)}, should be [0, 1]")
                print(f"output_img_tensor min: {torch.min(output_img_tensor)}, max: {torch.max(output_img_tensor)}, should be [0, 1]")

                # Compute epipolar consistency loss
                loss, output_dict_consistency_loss = consistency_loss(input_dict, F_i, refimg_tensor_rescaled,
                                                                      output_img_tensor_rescaled)


                loss_consistency = output_dict_consistency_loss['loss_consistency']
                loss_rgb = output_dict_consistency_loss['loss_rgb']


                if i_opt == 0:
                    wandb.log({"Original Consistency Loss": loss_consistency})
                    best_loss = loss
                    best_consistency_loss = loss_consistency
                    best_rgb_loss = loss_rgb
                    best_output_img = output_img
                    best_i_opt = i_opt
                    consistency_loss_megascenes = loss_consistency
                elif loss_consistency < best_consistency_loss:
                    best_loss = loss
                    best_consistency_loss = loss_consistency
                    best_rgb_loss = loss_rgb
                    best_output_img = output_img
                    best_i_opt = i_opt

                if not bEvaluateWarping: # Don't perform optimization if evaluating warping
                    print(f"++++++++++++++++++++++++++++++++++ Performing backwards step {i_opt} ++++++++++++++++++++++++++++++++++")
                    loss.backward()
                    if z_T.grad is not None:
                        print("z_T grad norm: ", z_T.grad.sum().item())
                        print("z_T.grad max: ", torch.max(z_T.grad))
                        print("z_T.grad min: ", torch.min(z_T.grad))

                    optimizer.step()
                    run_current_exp.log({"z_T norm": torch.norm(z_T)})
                    run_current_exp.log({"Change in z_T": torch.norm(z_T - z_T_start)})


                # Evaluate and log results (epipolar consistency loss, SEDs, keypoints, matches, etc.)
                good_samples = output_dict_consistency_loss['good_samples']
                SEDs = output_dict_consistency_loss['SEDs']
                if SEDs is not None:
                    e_line_im_1 = output_dict_consistency_loss['e_line_im_1']
                    e_line_im_2 = output_dict_consistency_loss['e_line_im_2']
                    closest_point_im1 = output_dict_consistency_loss['closest_point_im1']
                    closest_point_im2 = output_dict_consistency_loss['closest_point_im2']
                    kpts1 = output_dict_consistency_loss['kpts1']
                    kpts2 = output_dict_consistency_loss['kpts2']
                    print(f"------------------- Consistency Loss, i_opt={i_opt} -------------------")
                    print("Epipolar Consistency Loss: ", loss_consistency)
                    print("i: ", i)
                    if z_T.grad is not None:
                        print("z_T grad norm: ", z_T.grad.sum().item())
                    print("Change in z_T: ", torch.norm(z_T - z_T_start))
                    print("lr: ", lr)
                    print("Loss: ", loss)

                    percentage_within_10 = torch.sum(SEDs < 10) / SEDs.shape[0]
                    percentage_within_5 = torch.sum(SEDs < 5) / SEDs.shape[0]
                    percentage_within_2 = torch.sum(SEDs < 2) / SEDs.shape[0]
                    percentage_within_1 = torch.sum(SEDs < 1) / SEDs.shape[0]
                    print("Percentage of SEDs within 10: ", percentage_within_10)
                    print("Percentage of SEDs within 5: ", percentage_within_5)
                    print("Percentage of SEDs within 2: ", percentage_within_2)
                    print("Percentage of SEDs within 1: ", percentage_within_1)

                    #Create and save histogram of SEDs
                    SEDs_np = SEDs.cpu().detach().numpy()
                    np.save(join(histogram_folder, f"{i_opt}.npy"), SEDs_np)
                    plt.clf()
                    plt.hist(SEDs_np, bins=100)
                    plt.title("SEDs Histogram")
                    plt.xlabel("SEDs")
                    plt.ylabel("Frequency")
                    plt.xlim(0, 10)
                    plt.ylim(0,10000)
                    plt.savefig(histogram_folder + f"/{i_opt}.png")

                    if num_poses == 1:
                        plt.savefig(analysis_folder + f"/histogram.png")
                    plt.close()

                    log_dict = {
                        current_View + "/PercentageWithin1pixel": percentage_within_1,
                        current_View + "/PercentageWithin2pixels": percentage_within_2,
                        current_View + "/PercentageWithin5pixels": percentage_within_5,
                        current_View + "/PercentageWithin10pixels": percentage_within_10,
                        current_View + "/NumberMatches": num_matches,
                        current_View + "/loss": loss,
                        current_View + "/consistency_loss": loss_consistency,
                        current_View + "/rgb_loss": loss_rgb,
                        current_View + "/step": i_opt,
                    }
                    run_current_exp.log(log_dict)

                    if loss_rgb is not None:
                        run_current_exp.log({"RGB Loss": loss_rgb})

                    if bEvaluateWarping:
                        opt_image_path = warped_img_save_path # TODO Uncomment if want to use evaluate against warped image
                    num_matches_all = kpts1.shape[0]
                    keypoints_img = draw_matches(kpts1, kpts2, refimg_path, opt_image_path,
                                               num_matches_all, draw_lines=False, random_sampling=False)
                    keypoints_img.save(keypoints_folder + f"/{i_opt}.png")


                    num_matches_show = 5
                    matches_img = draw_matches(kpts1, kpts2, refimg_path, opt_image_path,
                                               num_matches_show, draw_lines=True, random_sampling=False)
                    matches_img.save(matches_folder + f"/{i_opt}.png")


                    if (not (e_line_im_2[0,0] == 0 and e_line_im_2[0,1] == 0 and e_line_im_2[0,2] == 0)) and (not (e_line_im_1[0,0] == 0 and e_line_im_1[0,1] == 0 and e_line_im_1[0,2] == 0)):

                        num_epishow = 10
                        for i_show in range(num_epishow):
                            random.seed(i_show) # Setting seed just for random sampling in visualization
                            epi_both_img_no_matches = draw_matches(kpts1, kpts2, refimg_path, opt_image_path,
                                num_matches_show, draw_lines=False, random_sampling=True, closest_point_im1=closest_point_im1,
                                e_line_im_1=e_line_im_1, closest_point_im2= closest_point_im2, e_line_im_2=e_line_im_2, SEDs = SEDs)
                            random.seed(seed)

                            epipolar_both_no_matches_folder_i = os.path.join(epipolar_both_no_matches_folder, f"{i_show}")
                            create_folder_if_not_exists(epipolar_both_no_matches_folder_i)
                            epi_both_img_no_matches.save(epipolar_both_no_matches_folder_i + f"/{i_opt}.png")
                            if i_opt==0:
                                epi_both_img_no_matches.save(epipolar_both_no_matches_folder + f"/{i_show}.png")




            wandb.log({"Best Consistency Loss": best_consistency_loss})
            log_dict = {
                current_View + "/BestLoss": best_loss,
                current_View + "/BestRGBLoss": best_rgb_loss,
                current_View + "/BestConsistencyLoss": best_consistency_loss,
                current_View + "/Best i_opt": best_i_opt,
                current_View + "/step": i_opt,
            }
            # Save best_i_opt to text file
            best_i_opt_path = join(args.savepath,f"{current_azimuth}", "best_i_opt.txt")
            with open(best_i_opt_path, "w") as f:
                f.write(str(best_i_opt))
            run_current_exp.log(log_dict)


            if consistency_loss_megascenes is not None and best_consistency_loss is not None:
                improvement_consistency_loss = consistency_loss_megascenes - best_consistency_loss
                run_current_exp.log({current_View + "/Improvement in consistency loss": improvement_consistency_loss, current_View + "/step": i_opt})
                # Save improvement in consistency loss to textfile
                improvement_consistency_loss_path = join(args.savepath,f"{current_azimuth}", "improvement_consistency_loss.txt")
                original_consistency_loss_path = join(args.savepath,f"{current_azimuth}", "original_consistency_loss.txt")
                opt_consistency_loss_path = join(args.savepath,f"{current_azimuth}", "opt_consistency_loss.txt")
                with open(improvement_consistency_loss_path, "w") as f:
                    f.write(str(improvement_consistency_loss.item()))
                with open(original_consistency_loss_path, "w") as f:
                    f.write(str(consistency_loss_megascenes.item()))
                with open(opt_consistency_loss_path, "w") as f:
                    f.write(str(best_consistency_loss.item()))

            gen_sequence_path = optimized_sequence_folder + f"{current_azimuth}.png"
            gen_sequence_path_reference = optimized_sequence_folder + f"0_ref.png"
            gen_sequence_path_direct = os.path.join(args.savepath,f"{current_azimuth}_opt.png")



            Image.fromarray(best_output_img[0]).save(gen_sequence_path)
            Image.fromarray(best_output_img[0]).save(gen_sequence_path_direct)
            Image.fromarray(refimg_nopad).save(gen_sequence_path_reference)

            both_sequence_path_ref = both_sequence_folder + f"0_ref.png"
            all_both_sequence_path_ref = all_sequence_folder + f"{scene_name}_0.png"
            both_img_ref = np.concatenate((refimg_nopad, refimg_nopad, refimg_nopad), axis=0)
            Image.fromarray(both_img_ref).save(both_sequence_path_ref)
            Image.fromarray(both_img_ref).save(all_both_sequence_path_ref)

            both_sequence_path = both_sequence_folder + f"{current_azimuth}.png"
            all_both_sequence_path = all_sequence_folder + f"{scene_name}_{current_azimuth}.png"

            both_img = np.concatenate((warpedimg, original_output_img, best_output_img[0]), axis=0)
            Image.fromarray(both_img).save(both_sequence_path)
            Image.fromarray(both_img).save(all_both_sequence_path)

            run_current_exp.log({current_View + "/Best Optimized image": wandb.Image(gen_sequence_path), current_View + "/step": i_opt})


            newdataloader = {}

    print("******** Done with generating orbit sequence! ********")
    print("outputs length: ", len(outputs))
    print("outputs[0].shape: ", outputs[0].shape)

    results_dict = {"best_loss": best_loss, "best_rgb_loss": best_rgb_loss,
                    "best_consistency_loss": best_consistency_loss,
                    "improvement_consistency_loss": improvement_consistency_loss,
                    "best_i_opt": best_i_opt}

    return results_dict




if __name__ == '__main__':
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e", required=True,
        help="Directory for logging. Should include 'specs.yaml'",
    )
    arg_parser.add_argument(
        "--resume", "-r", required=True, type=int,
        help="continue from previous saved logs, integer value",
    )

    arg_parser.add_argument("--debug", "-d", action='store_true')

    arg_parser.add_argument("--savepath", "-s", required=True)
    arg_parser.add_argument("--inputimg", "-i", required=True)
    
    arg_parser.add_argument("--zeronvs", "-z", action='store_true')
    arg_parser.add_argument("--warponly", "-w", action='store_true')

    arg_parser.add_argument("--ckpt_file", action='store_true', help='if checkpoint file is .ckpt instead of safetensors')

    arg_parser.add_argument("--intrinsics_path", default='default', type=str, help='path to intrinsics')


    arg_parser.add_argument("--batch_size", "-b", default=9, type=int, help='effective batch size is batch_size*repeat; lower either if OOM')
    arg_parser.add_argument("--repeat", default=10, type=int, help='number of generations for each camera position')

    arg_parser.add_argument("--cfg", default=3, type=float, help='cfg scale')

    arg_parser.add_argument("--num_poses", default=1, type=int, help='number of poses to generate')
    arg_parser.add_argument("--x_end", default=0.2, type=float, help='x end for orbit')
    arg_parser.add_argument("--rotation_angle", default=10, type=float, help='rotation angle for orbit')
    arg_parser.add_argument("--exp_name", default='default', type=str, help='experiment name')

    arg_parser.add_argument("--lr", default=0.1, type=float, help='learning rate for ug')
    arg_parser.add_argument("--num_opt_steps", default=5, type=int, help='number of optimization steps for ug')
    arg_parser.add_argument("--alpha_rgb", default=1, type=float, help='alpha for rgb loss')

    arg_parser.add_argument("--seed", default=0, type=int, help='seed for random number generator')

    arg_parser.add_argument("--input_image_idx", default=0, type=int, help='index of input image')
    arg_parser.add_argument("--num_matches", default=5000, type=int, help='number of matches')
    arg_parser.add_argument("--ddim_steps", default=50, type=int, help='number of ddim steps')

    arg_parser.add_argument("--scene_name", default='apple1', type=str, help='experiment name')
    arg_parser.add_argument("--split_type", default='validation', type=str, help='split type (test/validation)')

    arg_parser.add_argument("--certainty_threshold", default=0.25, type=float, help='certainty threshold for matches')
    arg_parser.add_argument("--az_start", default=5, type=float, help='start_azimuth_value')
    arg_parser.add_argument("--az_end", default=25, type=float, help='end_azimuth_value')
    arg_parser.add_argument("--az_step", default=5, type=float, help='azimuth_step')
    args = arg_parser.parse_args()
    print(args)

    print(args.savepath)

    # Check if args.savepath exists and if it does then cancels script
    if os.path.exists(args.savepath):
        print("Savepath already exists. Exiting script.")
        sys.exit()
    print("resume: ", args.resume)
    print("batch_size: ", args.batch_size)
    print("repeat: ", args.repeat)
    results_dict = run_opt(args)