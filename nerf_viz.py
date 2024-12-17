import os
import torch
import math
import imageio
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from pathlib import Path

from utils_nerf_viz import Rays, render_image, _load_renderings
from nerfacc import ContractionType, OccupancyGrid
from instant_ngp import NGPradianceField 

parser = argparse.ArgumentParser(description="Create video representation of a 3D object, using the neural radiance field weights.")

parser.add_argument('--directory_path', type= str, required= True, help = 'Path to the directory in which you want the data you need represented')
parser.add_argument('--max_images', type= int, default= 120, help= 'The number of images you to iterate over for rendering')
parser.add_argument('--fps', type= int, default= 30, help= 'the frame rate per second of the output video')

args = parser.parse_args()

def get_translation_t(t):
    """Get the translation matrix for movement in t."""
    matrix = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1]
    ]
    return torch.tensor(matrix, dtype=torch.float32)


def get_rotation_phi(phi):
    """Get the rotation matrix for movement in phi."""
    matrix = [
        [1, 0, 0, 0],
        [0, torch.cos(phi), -torch.sin(phi), 0],
        [0, torch.sin(phi), torch.cos(phi), 0],
        [0, 0, 0, 1]
    ]
    return torch.tensor(matrix, dtype=torch.float32)


def get_rotation_theta(theta):
    """Get the rotation matrix for movement in theta."""
    matrix = [
        [torch.cos(theta), 0, -torch.sin(theta), 0],
        [0, 1, 0, 0],
        [torch.sin(theta), 0, torch.cos(theta), 0],
        [0, 0, 0, 1]
    ]
    return torch.tensor(matrix, dtype=torch.float32)


def pose_spherical(theta, phi, t):
    """
    Get the camera to world matrix for the corresponding theta, phi
    and t.
    """
    c2w = get_translation_t(t)
    c2w = get_rotation_phi(phi / 180.0 * np.pi) @ c2w
    c2w = get_rotation_theta(theta / 180.0 * np.pi) @ c2w
    c2w = torch.from_numpy(np.array([
        [-1, 0, 0, 0], 
        [ 0, 0, 1, 0], 
        [ 0, 1, 0, 0], 
        [ 0, 0, 0, 1]
    ], dtype=np.float32)) @ c2w  
    
    return c2w


def create_video(
    width,
    height,
    device,
    focal,
    radiance_field,
    occupancy_grid,
    scene_aabb,
    near_plane,
    far_plane,
    render_step_size,
    render_bkgd,
    cone_angle,
    alpha_thre,
    path,
    OPENGL_CAMERA=True):

    rgb_frames = []

    # Iterate over different theta value and generate scenes.

    max_images = args.max_images
    array = np.linspace(-30.0, 30.0, max_images//2, endpoint=False)
    array = np.append(array, np.linspace(30.0, -30.0, max_images//2, endpoint=False))
    
    for index, theta in tqdm(enumerate(np.linspace(0.0, 360.0, max_images, endpoint=False))):

        # Get the camera to world matrix.
        c2w = pose_spherical(torch.tensor(theta), torch.tensor(array[index]), torch.tensor(1.0))
        c2w = c2w.to(device)

        x, y = torch.meshgrid(
            torch.arange(width, device=device),
            torch.arange(height, device=device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()

        K = torch.tensor([
            [focal, 0, width / 2.0],
            [0, focal, height / 2.0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)  # (3, 3)

        camera_dirs = F.pad(
            torch.stack([
                (x - K[0, 2] + 0.5) / K[0, 0],
                (y - K[1, 2] + 0.5) / K[1, 1] * (-1.0 if OPENGL_CAMERA else 1.0),
            ], dim=-1),
            (0, 1),
            value=(-1.0 if OPENGL_CAMERA else 1.0)
        )  # [num_rays, 3]
        camera_dirs.to(device)

        directions = (camera_dirs[:, None, :] * c2w[:3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(directions, dim=-1, keepdims=True)

        origins = torch.reshape(origins, (height, width, 3))
        viewdirs = torch.reshape(viewdirs, (height, width, 3))
        
        rays = Rays(origins=origins, viewdirs=viewdirs)
        # render
        rgb, _, _, _ = render_image(
            radiance_field=radiance_field,
            occupancy_grid=occupancy_grid,
            rays=rays,
            scene_aabb=scene_aabb,
            # rendering options
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )

        numpy_image = (rgb.cpu().numpy() * 255).astype(np.uint8)
        rgb_frames.append(numpy_image)

    #this should be input

    fps = args.fps
    imageio.mimwrite(path, rgb_frames, fps=fps, quality=8, macro_block_size=None)
    
    #converting mp4 to gif
    reader = imageio.get_reader(video_path_mp4)
    writer = imageio.get_writer(video_path_gif, fps=30) # Set the frames per second

    for frame in reader:
        writer.append_data(frame)

    writer.close()
    

if __name__ == "__main__":

    directory_path = args.directory_path
    split = "train" 

    parent_dir = os.path.basename(os.path.dirname(directory_path)) 
    subdir = os.path.basename(directory_path)
    subject_id = f"{parent_dir}/{subdir}"

    n_hidden_layers = 3
    n_neurons = 64
    coordinate_encoding = "Frequency"
    encoding_size = 24
    mlp = "FullyFusedMLP"
    activation = "ReLU"
    
    data_root = "shapenet_render"
    video_path_mp4 = Path(directory_path) / "nerf.mp4"
    video_path_gif = Path(directory_path) / "nerf.gif"
    device = "cuda"

    aabb = [-0.7,-0.7,-0.7,0.7,0.7,0.7]
    scene_aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
    alpha_thre = 0.0
    cone_angle = 0.0
    contraction_type = ContractionType.AABB
    far_plane = None
    grid_resolution = 96
    near_plane = None
    render_n_samples = 1024
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / render_n_samples
    ).item()
    target_sample_batch_size = 1 << 18
    unbounded = False

    images, camtoworlds, focal = _load_renderings(directory_path ,split)

    radiance_field = NGPradianceField(
        aabb=aabb,
        unbounded=unbounded,
        encoding=coordinate_encoding,
        mlp=mlp,
        activation=activation,
        n_hidden_layers=n_hidden_layers,
        n_neurons=n_neurons,
        encoding_size=encoding_size
    ).to(device)
    radiance_field.load_state_dict(torch.load(Path(directory_path)/ "nerf_weights.pth"))
    radiance_field = radiance_field.eval()


    occupancy_grid = OccupancyGrid(
        roi_aabb=aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
    ).to(device)
    checkpoint = torch.load(Path(directory_path) / "grid.pth")
    if "_binary" in checkpoint and checkpoint["_binary"].is_sparse:
        checkpoint["_binary"] = checkpoint["_binary"].to_dense()
    occupancy_grid.load_state_dict(checkpoint, strict= False)
    occupancy_grid = occupancy_grid.eval()

    with torch.no_grad():
        create_video(
            720, 
            480, 
            device, 
            focal, 
            radiance_field, 
            occupancy_grid, 
            scene_aabb,
            near_plane, 
            far_plane, 
            render_step_size,
            render_bkgd= torch.zeros(3, device=device),
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
            path=video_path_mp4
        )