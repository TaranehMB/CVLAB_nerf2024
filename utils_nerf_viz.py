import random
from typing import Optional
import os
import imageio.v2 as imageio
import json
import concurrent

import numpy as np
import torch
import collections
import torch.nn.functional as F
from typing import Any, Dict  
import collections 
from nerfacc import OccupancyGrid, ray_marching, rendering

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))


def unflatten_mlp_params(
    params: torch.Tensor,
    sample_sd: Dict[str, Any],
) -> Dict[str, Any]:
    sd = collections.OrderedDict()

    start = 0
    for k in sample_sd:
        end = start + sample_sd[k].numel()
        layer_params = params[start:end].view(sample_sd[k].shape)
        sd[k] = layer_params
        start = end

    return sd


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
 
def sample_pixels_uniformly(opacities, colors, max_elements):
    # Get indices of True and False elements, where True means that the specified coordinate 
    # contains the 3d model, False otherwise.

    true_indices =  torch.nonzero(opacities.squeeze()).squeeze()
    if len(true_indices) < max_elements:
        print('true_indices < max_elements')
        n_missing_elements = max_elements - len(true_indices)
        
        true_indices_pad = true_indices[-n_missing_elements:]
        true_indices = torch.cat((true_indices, true_indices_pad), dim=0)

    merged_indices = true_indices[:max_elements]
    new_colors = colors[merged_indices]
    new_opacities = opacities[merged_indices]

    return new_colors, new_opacities, merged_indices

#loader for loading renderings

def read_image(file_path):
    return imageio.imread(file_path)

def _load_renderings(data_dir: str, split: str):
    
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    file_paths = []
    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        
        file_paths.append(fname)

        camtoworlds.append(frame["transform_matrix"])
        
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(read_image, file_paths)
        images = list(results)
    
    
    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal

#image_rendering

def render_image(
    # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccupancyGrid,
    rays: Rays,
    scene_aabb: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    # only useful for dnerf
    timestamps: Optional[torch.Tensor] = None,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field.query_density(positions, t)
        
        _, density = radiance_field._query_density_and_rgb(positions, None)
        return density

        # ORIGINAL
        # return radiance_field.query_density(positions)
        

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if timestamps is not None:
            # dnerf
            t = (
                timestamps[ray_indices]
                if radiance_field.training
                else timestamps.expand_as(positions[:, :1])
            )
            return radiance_field(positions, t, t_dirs)
        return radiance_field(positions, t_dirs)

    results = []
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = ray_marching(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            grid=occupancy_grid,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )
        rgb, opacity, depth = rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=render_bkgd,
        )
        chunk_results = [rgb, opacity, depth, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        sum(n_rendering_samples),
    )