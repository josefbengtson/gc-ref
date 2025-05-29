import shutil
from pathlib import Path
import enlighten
import pycolmap
from pycolmap import logging, ImageReaderOptions
import numpy as np
import open3d as o3d
import os
import time

def set_focal_length(database_path, image_name, f_x, f_y):
    db = pycolmap.Database(database_path)
    image = db.read_image(image_name)
    camera_id = image.camera_id
    camera = db.read_camera(camera_id)
    camera.params[0] = f_x
    camera.params[1] = f_y
    db.update_camera(camera)
    db.close()

def create_reconstruction(output_path, K, max_error):
    print("output_path: ", output_path)
    image_path = output_path / "images"
    database_path = output_path / "database.db"
    sfm_path = output_path / "sfm"

    output_path.mkdir(exist_ok=True)
    # The log filename is postfixed with the execution timestamp.
    logging.set_log_destination(logging.INFO, output_path / "INFO.log.")
    assert image_path.exists(), "Image path does not exist!"
    if database_path.exists():
        database_path.unlink()
    pycolmap.set_random_seed(0)


    pycolmap.extract_features(database_path, image_path, camera_model="PINHOLE")

    f_x = K[0,0]
    f_y = K[1,1]
    for image_file in image_path.iterdir():
        set_focal_length(database_path, image_file.name, f_x, f_y)

    pycolmap.match_exhaustive(database_path)
    # Show pycolomap image ids
    if sfm_path.exists():
        shutil.rmtree(sfm_path)
    sfm_path.mkdir(exist_ok=True)

    recs = incremental_mapping_with_pbar(database_path, image_path, sfm_path, max_error)
    for idx, rec in recs.items():
        logging.info(f"#{idx} {rec.summary()}")

def get_camera_lineset(camera,K, im_size, cam_scale=0.1):
    # cam_scale = 0.05
    R = camera[:3,:3]
    t = camera[:3,3]
    T = np.zeros((4, 4))
    T[:3, :3] = R # Since we multiply from the right, not left!
#     T[:3, :3] = R.T # Since we multiply from the right, not left!
    T[3, 3] = 1
    T[:3, 3] = t
    cam_vis = o3d.geometry.LineSet.create_camera_visualization(
        view_width_px=im_size[1],
        view_height_px=im_size[0],
        extrinsic=T,
        intrinsic=K,
        scale=1*cam_scale
    )
    return cam_vis


def load_reconstruction(reconstruction):
    # Load camera calibration matrix
    for camera_id, camera in reconstruction.cameras.items():
        K = camera.calibration_matrix()



    # Extract 3D points
    points = []
    for point3D_id, point3D in reconstruction.points3D.items():
        points.append(point3D.xyz)
    # Convert to numpy arrays
    X = np.array(points)
    np.set_printoptions(suppress=True, precision=3)
    T0 = None
    for image_id, image in reconstruction.images.items():
        pose = image.cam_from_world.matrix()
        # Extract keypoints
        # assert False, "Check image"
        if image.name == "reference.png":
            pose0 = pose
            T0 = np.vstack([pose0, np.array([0, 0, 0, 1])])

    if T0 is None:
        raise ValueError("T0 is None")

    for image_id, image in reconstruction.images.items():

        if image.name == "reference.png":
            continue
        pose = image.cam_from_world.matrix()

        # Transform pose to relative pose
        T = np.vstack([pose, np.array([0, 0, 0, 1])])
        T_relative =   T @ np.linalg.inv(T0)
        P_rel = T_relative[:3,:]
    return P_rel, T0, K, X

def extract_normalized_translation(pose):
    t = pose[:3,3]
    t_normalized = t/np.linalg.norm(t)
    return t_normalized

def normalize_pose_translation(pose,scale=1):
    t = pose[:3,3]
    t_normalized = (scale*t)/np.linalg.norm(t)
    pose[:3,3] = t_normalized
    return pose

def eval_pose_accuracy(P1, P2):
    # compute angle between t1 and t2
    t1 = P1[:3,3]
    t2 = P2[:3,3]
    t_angle_dist = np.arccos(np.dot(t1,t2)/(np.linalg.norm(t1)*np.linalg.norm(t2)))

    # compute
    t1_norm = extract_normalized_translation(P1)
    t2_norm = extract_normalized_translation(P2)
    t_dist = np.linalg.norm(t1_norm - t2_norm)

    R1 = P1[:3,:3]
    R2 = P2[:3,:3]

    rot_dist = np.arccos((np.trace(np.dot(R1, R2.T))-1)/2)
    return t_dist, t_angle_dist, rot_dist

def compute_pose_accuracy(path_recon, pose_path = None, P_gt=None,  save_path="vis.png", visualize=False, K=None, max_error = 4, P_vis=None):
    if not Path(os.path.join(path_recon,"sfm/0")).exists(): # Check if reconstruction exists at path
        print("Performing reconstruction at path: ", path_recon)
        create_reconstruction(Path(path_recon), K, max_error)
    else:
        print("Reconstruction already exists at path: ", path_recon)

    assert K is not None, "K is None in compute_pose_accuracy"
    # Load P_gt
    if pose_path is not None:
        poses = np.load(pose_path)
        P_gt = poses[1]
        P_gt = np.linalg.inv(P_gt)
    elif P_gt is None:
        raise ValueError("P_gt is None")

    if Path(os.path.join(path_recon, "sfm/0")).exists():
        print("Loading reconstruction")
        reconstruction = pycolmap.Reconstruction(f"{path_recon}/sfm/0")
        P_gen, P0, K_colmap, X = load_reconstruction(reconstruction)
        assert np.allclose(K, K_colmap), "K and K_colmap are not equal"


        # Compute translation and rotation distances
        t_dist, t_angle_dist, rot_dist = eval_pose_accuracy(P_gt, P_gen)
        rot_dist_deg = np.rad2deg(rot_dist)
        t_angle_dist_deg = np.rad2deg(t_angle_dist)

        # Visualize camera translations before visualization
        P_gt = normalize_pose_translation(P_gt)
        P_gen = normalize_pose_translation(P_gen)

        if visualize:
            # Create an offscreen renderer
            WIDTH = 800
            HEIGHT = 800
            renderer = o3d.visualization.rendering.OffscreenRenderer(WIDTH, HEIGHT)


            # Load camera parameters
            image_shape = [int(K[0, 2] * 2), int(K[1, 2] * 2)]
            cam_size = 0.5

            # Add input camera
            P_input = np.eye(4)
            input_camera = get_camera_lineset(P_input, K, image_shape, cam_size)
            input_camera.paint_uniform_color(np.array([[0.], [0.], [1.]]))
            renderer.scene.add_geometry("input_camera", input_camera, o3d.visualization.rendering.MaterialRecord())

            # Add ground truth camera
            gt_camera = get_camera_lineset(P_gt,K,image_shape, cam_size)
            gt_camera.paint_uniform_color(np.array([[0.], [0.], [0.]]))
            renderer.scene.add_geometry("gt_camera", gt_camera, o3d.visualization.rendering.MaterialRecord())

            # Add generated camera
            cam_vis_rel = get_camera_lineset(P_gen,K,image_shape, cam_size)
            cam_vis_rel.paint_uniform_color(np.array([[0.], [1.], [0.]]))
            renderer.scene.add_geometry(f"cam_vis_ms_{id(cam_vis_rel)}", cam_vis_rel, o3d.visualization.rendering.MaterialRecord())
            # Add camera before optimization
            if P_vis is not None:
                cam_vis_vis = get_camera_lineset(P_vis, K, image_shape, cam_size)
                cam_vis_vis.paint_uniform_color(np.array([[1.], [0.], [0.]]))
                renderer.scene.add_geometry(f"cam_vis_vis_{id(cam_vis_vis)}", cam_vis_vis,
                                            o3d.visualization.rendering.MaterialRecord())

            center = -P_gt[:3,3]
            eye = [2, 2, 2]
            up = [1, 0, 0]
            renderer.scene.camera.look_at(center, eye, up)

            # Render the scene and save the image
            image = renderer.render_to_image()
            o3d.io.write_image(save_path, image)
    else:
        print("COLMAP Reconstruction not succesful!")
        t_dist = None
        t_angle_dist_deg = None
        rot_dist_deg = None
        P_gen = None
        P0 = None
        K = None
        X = None

    return t_dist, t_angle_dist_deg, rot_dist_deg, P_gt, P_gen, P0, K, X

def incremental_mapping_with_pbar(database_path, image_path, sfm_path, max_error):
    num_images = pycolmap.Database(database_path).num_images
    options =  pycolmap.IncrementalPipelineOptions()
    options.init_image_id1 = 1
    options.init_image_id2 = 2
    options.ba_refine_focal_length = False
    options_incremental_mapper = pycolmap.IncrementalMapperOptions()
    options_incremental_mapper.init_max_error = max_error
    options.mapper = options_incremental_mapper
    with enlighten.Manager() as manager:
        with manager.counter(
            total=num_images, desc="Images registered:"
        ) as pbar:
            pbar.update(0, force=True)

            print("Starting incremental mapping")
            reconstructions = pycolmap.incremental_mapping(
                database_path,
                image_path,
                sfm_path,
                initial_image_pair_callback=lambda: pbar.update(2),
                next_image_callback=lambda: pbar.update(1),
                options=options

            )
            print("Incremental mapping done")
    return reconstructions