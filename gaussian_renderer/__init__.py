#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def filter_gaussians_for_tile(means2D, tile_x, tile_y, tile_size, image_width, image_height):
    """ Lọc Gaussian nằm trong một tile cụ thể """
    x_min, x_max = tile_x * tile_size, (tile_x + 1) * tile_size
    y_min, y_max = tile_y * tile_size, (tile_y + 1) * tile_size

    inside_tile = (
        (means2D[:, 0] >= x_min) & (means2D[:, 0] < x_max) &
        (means2D[:, 1] >= y_min) & (means2D[:, 1] < y_max)
    )
    return inside_tile

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False, resolution=None):

    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if resolution is not None:
        image_height, image_width = resolution
    else:
        image_height, image_width = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        # Kích thước tile (có thể điều chỉnh, thử nghiệm với 128, 64, 32)
        tile_size = 128  
        num_tiles_x = image_width // tile_size
        num_tiles_y = image_height // tile_size
        
        # Tạo ảnh trống để chứa kết quả render
        rendered_image = torch.zeros((3, image_height, image_width), device="cuda")
        depth_image = torch.zeros((image_height, image_width), device="cuda")
        radii = torch.zeros((means2D.shape[0],), device="cuda")
        
        # Duyệt qua từng tile để render
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                # Lọc Gaussian thuộc về tile này
                tile_mask = filter_gaussians_for_tile(means2D, i, j, tile_size, image_width, image_height)
                if not tile_mask.any():
                    continue  # Bỏ qua tile nếu không có Gaussian bên trong
        
                # Lấy Gaussian của tile
                tile_means2D = means2D[tile_mask]
                tile_means3D = means3D[tile_mask]
                tile_opacity = opacity[tile_mask]
                tile_scales = scales[tile_mask] if scales is not None else None
                tile_rotations = rotations[tile_mask] if rotations is not None else None
                tile_cov3D_precomp = cov3D_precomp[tile_mask] if cov3D_precomp is not None else None
                tile_shs = shs[tile_mask] if shs is not None else None
                tile_colors_precomp = colors_precomp[tile_mask] if colors_precomp is not None else None
        
                # Render tile
                tile_render, tile_radii, tile_depth = rasterizer(
                    means3D=tile_means3D,
                    means2D=tile_means2D,
                    shs=tile_shs,
                    colors_precomp=tile_colors_precomp,
                    opacities=tile_opacity,
                    scales=tile_scales,
                    rotations=tile_rotations,
                    cov3D_precomp=tile_cov3D_precomp
                )
        
                # Gán vào ảnh tổng
                rendered_image[:, i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size] += tile_render
                depth_image[i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size] += tile_depth
                radii[tile_mask] = tile_radii

        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
