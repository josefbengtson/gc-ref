import numpy as np 
import os, json, sys
import matplotlib.pyplot as plt

import DepthAnything.depth_anything.dpt
from DepthAnything.depth_anything.dpt import DepthAnything
from DepthAnything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2
import warnings

from PIL import Image
import torch

def get_x_poses(num_steps=6, radius=1, endpoint=2, w2c=True):
    extrinsics = []

    for step in range(num_steps):

        x = -endpoint + endpoint*2/(num_steps-1)*step
        t = np.array([x, 0, 0]) # z=radius
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, 3] = -t
        if not w2c:
            extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
        extrinsics.append(extrinsic_matrix)
        

    return extrinsics

def get_poses(num_steps=10, radius=1, rotation_angle=30, zscale=1, xscale=1, c2w=True, endpoint=(0,-1), direction='ccw'):
    extrinsics = []
    a,b = endpoint
    for step in range(num_steps):
        if direction == 'ccw':
            theta = np.radians(rotation_angle) * step / num_steps
        elif direction == 'cw':
            theta = -np.radians(rotation_angle) * step / num_steps
        else:
            raise ValueError("Direction must be either 'ccw' or 'cw'")

        # Rotation matrix (around the y-axis)
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Translation vector
        x = a + (b - a) * step / (num_steps-1)
        #x = radius *xscale* np.cos(theta)
        t = np.array([x, 0, radius * zscale * np.sin(theta)])
        #print(t)

        # Create extrinsic matrix (4x4)
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = t # -R @ t

        if c2w:
            extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
        extrinsics.append(extrinsic_matrix)
    return extrinsics


def get_poses_double(x_end = 0.1, radius=1, zscale=1, c2w=True, endpoint=(0, -1)):
    extrinsics = []

    num_steps = 3
    xs = [0, x_end, -x_end]
    for step in range(num_steps):
        # Translation vector
        x = xs[step]
        print("ccw: x for step ", step, ": ", x)
        t = np.array([x, 0, 0])

        # Rotation matrix
        R = np.eye(3)

        # Create extrinsic matrix (4x4)
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = t  # -R @ t

        if c2w:
            extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
        extrinsics.append(extrinsic_matrix)
    return extrinsics

def get_ccw_poses_original(num_steps=10, radius=1, rotation_angle=30, zscale=1, xscale=1, c2w=True, endpoint=(0, -1)):
    extrinsics = []
    a, b = endpoint
    # if num_steps == 1:
    #     num_steps = 2
    #
    for step in range(num_steps):
        theta = np.radians(rotation_angle) * step / (num_steps-1)
        print(f"Theta for step {step}: ", theta)

        # Rotation matrix (around the y-axis)
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Translation vector
        x = a + (b - a) * step / (num_steps - 1)
        print("ccw: x for step ", step, ": ", x)
        # x = radius *xscale* np.cos(theta)
        t = np.array([x, 0, radius * zscale * np.sin(theta)])

        # print(t)

        # Create extrinsic matrix (4x4)
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = t  # -R @ t

        if c2w:
            extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
        extrinsics.append(extrinsic_matrix)
    return extrinsics


def get_cw_poses_original(num_steps=10, radius=1, rotation_angle=30, zscale=1, xscale=1, c2w=True, endpoint=(0, 1)):
    extrinsics = []
    a, b = endpoint
    for step in range(num_steps):
        # Negative theta for clockwise rotation
        theta = -np.radians(rotation_angle) * step / (num_steps-1)

        # Rotation matrix (around the y-axis)
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Translation vector
        x = a + (b - a) * step / (num_steps - 1)
        print("cw: x for step ", step, ": ", x)

        t = np.array([x, 0, radius * zscale * np.sin(theta)])

        # Create extrinsic matrix (4x4)
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = t  # -R @ t

        if c2w:
            extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
        extrinsics.append(extrinsic_matrix)
    return extrinsics

# def get_ccw_poses(num_steps=10, radius=1, rotation_angle=30, zscale=1, xscale=1, c2w=True, endpoint=(0,-1)):
#     extrinsics = []
#     a,b = endpoint
#     for step in range(num_steps):
#         theta = np.radians(rotation_angle) * step / num_steps
#
#         # Rotation matrix (around the y-axis)
#         R = np.array([
#             [np.cos(theta), 0, np.sin(theta)],
#             [0, 1, 0],
#             [-np.sin(theta), 0, np.cos(theta)]
#         ])
#
#         # Translation vector
#         x = a + (b - a) * step / (num_steps-1)
#         #x = radius *xscale* np.cos(theta)
#         t = np.array([x, 0, radius * zscale * np.sin(theta)])
#         #print(t)
#
#         # Create extrinsic matrix (4x4)
#         extrinsic_matrix = np.eye(4)
#         extrinsic_matrix[:3, :3] = R
#         extrinsic_matrix[:3, 3] = t # -R @ t
#
#         if c2w:
#             print("Inverting extrinsic matrix")
#             extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
#         else:
#             print("Not inverting extrinsic matrix")
#         extrinsics.append(extrinsic_matrix)
#     return extrinsics
#
# def get_cw_poses(num_steps=10, radius=1, rotation_angle=30, zscale=1, xscale=1, c2w=True, endpoint=(0,1)):
#     extrinsics = []
#     a, b = endpoint
#     for step in range(num_steps):
#         # Negative theta for clockwise rotation
#         theta = -np.radians(rotation_angle) * step / num_steps
#
#         # Rotation matrix (around the y-axis)
#         # R = np.array([
#         #     [np.cos(theta), 0, np.sin(theta)],
#         #     [0, 1, 0],
#         #     [-np.sin(theta), 0, np.cos(theta)]
#         # ])
#
#         # Rotation matrix (around the x-axis)
#         R = np.array([
#             [1, 0, 0],
#             [0, np.cos(theta), -np.sin(theta)],
#             [0, np.sin(theta), np.cos(theta)]
#         ])
#
#         # Translation vector
#         x = a + (b - a) * step / (num_steps - 1)
#
#         t = np.array([x, 0, radius * zscale * np.sin(theta)]) # x axis translation
#         # t = np.array([0, x, radius * zscale * np.sin(theta)]) # y axis translation
#
#
#         # Create extrinsic matrix (4x4)
#         extrinsic_matrix = np.eye(4)
#         extrinsic_matrix[:3, :3] = R
#         extrinsic_matrix[:3, 3] = t  # -R @ t
#
#         if c2w:
#             extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
#         extrinsics.append(extrinsic_matrix)
#     return extrinsics


def get_orbit_poses_original(num_poses=2, cwx=(0,1), x_end=1, rotation_angle=15):
    # orbitposes = []

    orbitposes = get_poses_double(x_end=x_end)
    #
    # # ccwx = (0,-0.5)
    # cwx = (0, x_end)
    # ccwx = (0, -x_end)
    # ccwposes = get_ccw_poses_original(endpoint=ccwx, num_steps=num_poses, rotation_angle=rotation_angle)
    # b_only_ccw = True
    # if b_only_ccw:
    #     orbitposes.extend(ccwposes)
    # else:
    #     cwposes = get_cw_poses_original(endpoint=cwx, num_steps=num_poses)
    #     cwposes.reverse()
    #     orbitposes.extend(cwposes[:-1])
    #     orbitposes.extend(ccwposes)
    return orbitposes

def get_orbit_poses(cwx=(0,1), ccwx=(0,-1)):
    print("*****Entering get_orbit_poses*****")
    orbitposes = []
    rotation_angle = 30
    x_distance = 1
    bBothDir = False
    ccwx = (0,-x_distance)
    print("Rotation angle: ", rotation_angle)
    #total_num_steps = 20
    #num_steps = int(total_num_steps/2)
    num_steps = 5
    if bBothDir:
        cwx = (0,x_distance)
        cwposes = get_poses(endpoint=cwx, rotation_angle=rotation_angle, num_steps=num_steps, direction='cw')
        cwposes.reverse()
        cwposes_min1 = cwposes[:-1]
        orbitposes.extend(cwposes_min1) # Skip backward traj

    ccwposes = get_poses(endpoint=ccwx, rotation_angle=rotation_angle, num_steps=num_steps, direction='ccw')
    orbitposes.extend(ccwposes)


    print("ccwposes len: ", len(ccwposes))

    orbitposes.extend(ccwposes)
    print("orbitposes length: ", len(orbitposes))
    print("*****Exiting get_orbit_poses*****")
    return orbitposes


def get_front_facing_trans(num_frames, max_trans=2.0, c2w=True, z_div=2.0):
    output_poses = []

    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) /4.0 #* 3.0 / 4.0
        z_trans = -max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames)) / z_div

        i_pose = np.concatenate([
            np.concatenate(
                [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
                # [np.eye(3), np.array([x_trans, 0., 0.])[:, np.newaxis]], axis=1),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
        ],axis=0)[np.newaxis, :, :][0]

        if c2w:
            i_pose = np.linalg.inv(i_pose)
        output_poses.append(i_pose)

    return output_poses
  

def load_depth_model():
    # load depth model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # encoder = 'vitl' # can also be 'vitb' or 'vitl'
        # depth_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).cuda().eval()

        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }
        encoder = 'vitl'  # or 'vitb', 'vits'
        depth_anything = DepthAnything(model_configs[encoder])
        chkpts_path = f"/mimer/NOBACKUP/groups/snic2022-6-266/josef/MegaScenes/DepthAnything/depth_anything_{encoder}14.pth"
        # If chkpts_path does not exist, try using the following path:
        if not os.path.exists(chkpts_path):
            chkpts_path = f"/home/josefoffice/Mimer2/MegaScenes/DepthAnything/depth_anything_{encoder}14.pth"

        depth_anything.load_state_dict(torch.load(chkpts_path))
        depth_model = depth_anything.cuda().eval()


        dtransform = Compose([
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
        return depth_model, dtransform

# def invert_depth(depth_map):
#     inv = depth_map.copy()
#     disparity_max = 1000
#     disparity_min = 0.001
#     inv[inv > disparity_max] = disparity_max
#     inv[inv < disparity_min] = disparity_min
#     inv = 1.0 / inv
#     return inv

def invert_depth(depth_map):
    inv = depth_map.clone()
    # disparity_max = 1000
    disparity_min = 0.001
    # inv[inv > disparity_max] = disparity_max
    inv[inv < disparity_min] = disparity_min
    inv = 1.0 / inv
    return inv

# def save_images_as_grid(imgs, fixed_height=256, spacing=5):
#     """
#     Save a grid of images with the same height and a spacing between them, expanding horizontally.

#     :param imgs: List of NumPy images
#     :param save_path: Path to save the image
#     :param fixed_height: Fixed height for each image in the grid
#     :param spacing: Space between images in pixels
#     """
#     total_width = 0
#     resized_images = []

#     # Resize each image and calculate total width with spacing
#     for idx, np_img in enumerate(imgs):
#         # if idx == len(imgs)-1 or idx==len(imgs)-2: # for saving depth maps 
#         #     depth_map = np_img
#         #     normalized_depth = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
#         #     scaled_depth = (255 * normalized_depth).astype(np.uint8)
#         #     img = Image.fromarray(scaled_depth, 'L')  # 'L' mode for grayscale
#         # else:
#         img = Image.fromarray(np_img)
#         aspect_ratio = img.width / img.height
#         new_width = int(fixed_height * aspect_ratio)
#         resized_img = img.resize((new_width, fixed_height))
#         resized_images.append(resized_img)
#         total_width += new_width + spacing

#     total_width -= spacing  # Remove extra spacing at the end

#     # Create a new blank image with a white background
#     grid_img = Image.new('RGB', (total_width, fixed_height), color='white')

#     # Paste each resized image into the grid with spacing
#     x_offset = 0
#     for img in resized_images:
#         grid_img.paste(img, (x_offset, 0))
#         x_offset += img.width + spacing

#     # Save the grid image
#     return grid_img #.save(save_path)

def save_images_as_grid(imgs, fixed_height=256, spacing=5, max_per_row=5):
    """
    Save a grid of images with a maximum number of images per row.

    :param imgs: List of NumPy images
    :param fixed_height: Fixed height for each image in the grid
    :param spacing: Space between images in pixels
    :param max_per_row: Maximum number of images per row
    """
    row_widths = []
    row_images = []
    current_row = []

    from PIL import Image
    # Process images and organize them into rows
    for np_img in imgs:
        img = Image.fromarray(np_img)
        aspect_ratio = img.width / img.height
        new_width = int(fixed_height * aspect_ratio)
        resized_img = img.resize((new_width, fixed_height))

        if len(current_row) < max_per_row:
            current_row.append(resized_img)
        else:
            row_widths.append(sum(img.width for img in current_row) + spacing * (len(current_row) - 1))
            row_images.append(current_row)
            current_row = [resized_img]

    # Add last row
    if current_row:
        row_widths.append(sum(img.width for img in current_row) + spacing * (len(current_row) - 1))
        row_images.append(current_row)

    total_width = max(row_widths)
    total_height = fixed_height * len(row_images) + spacing * (len(row_images) - 1)

    # Create a new blank image with a white background
    grid_img = Image.new('RGB', (total_width, total_height), color='white')

    # Paste each resized image into the grid
    y_offset = 0
    for row in row_images:
        x_offset = 0
        for img in row:
            grid_img.paste(img, (x_offset, y_offset))
            x_offset += img.width + spacing
        y_offset += fixed_height + spacing

    # Return the grid image
    return grid_img