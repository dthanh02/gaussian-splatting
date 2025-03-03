# Copyright (C) 2023, Inria + Enhancements
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.tensorboard import SummaryWriter
import lpips
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# Hàm Total Variation Loss
def tv_loss(img):
    h_var = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    w_var = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return h_var + w_var

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = SummaryWriter(dataset.model_path) if 'SummaryWriter' in globals() else None
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Khởi tạo optimizer AdamW và scaler cho mixed precision
    optimizer = optim.AdamW(gaussians.parameters(), lr=opt.learning_rate, weight_decay=1e-5)
    scaler = GradScaler()

    # Khởi tạo LPIPS
    lpips_loss_fn = lpips.LPIPS(net='alex').to("cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    progress_bar = tqdm(range(first_iter, 30000), desc="Training progress")  # Giới hạn tại 30,000 iterations

    lambda_tv = 0.00005  # Hệ số TV Loss
    lambda_lpips = 0.005  # Hệ số LPIPS Loss

    for iteration in range(first_iter + 1, 30001):
        gaussians.update_learning_rate(iteration)
        lambda_dssim = min(0.7, 0.4 + 0.3 * (iteration / 30000))  # Điều chỉnh động lambda_dssim

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Lấy viewpoint ngẫu nhiên
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render với mixed precision
        with autocast():
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()

            # Tính loss
            Ll1 = l1_loss(image, gt_image)
            ssim_value = ssim(image, gt_image)
            loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value)

            # Thêm TV Loss
            Ltv = tv_loss(image.unsqueeze(0))
            loss += lambda_tv * Ltv

            # Thêm LPIPS Loss mỗi 50 iteration
            if iteration % 50 == 0:
                Llpips = lpips_loss_fn(image.unsqueeze(0), gt_image.unsqueeze(0)).mean()
                loss += lambda_lpips * Llpips

        # Backward và tối ưu hóa với mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Cập nhật progress bar
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.7f}"})
            progress_bar.update(10)

        # Báo cáo và lưu kết quả
        if iteration in testing_iterations:
            psnr_test = evaluate_psnr(scene, render, pipe, background)
            print(f"[ITER {iteration}] PSNR: {psnr_test}")
            if tb_writer:
                tb_writer.add_scalar('test/psnr', psnr_test, iteration)

        if iteration in saving_iterations:
            scene.save(iteration)

    progress_bar.close()
    print("Training complete at 30,000 iterations.")

def evaluate_psnr(scene, render_func, pipe, background):
    psnr_test = 0.0
    test_cameras = scene.getTestCameras()
    for viewpoint in test_cameras:
        image = torch.clamp(render_func(viewpoint, scene.gaussians, pipe, background)["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
        psnr_test += psnr(image, gt_image).mean().double()
    return psnr_test / len(test_cameras)

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--test_iterations', nargs="+", type=int, default=[7000, 30000])
    parser.add_argument('--save_iterations', nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, [], args.checkpoint, -1)
