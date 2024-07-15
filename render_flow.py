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
import matplotlib.pyplot as plt
import heapq
import torch
import math
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, vis_depth,vis_depth1
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
from skimage.metrics import structural_similarity
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import cv2
from matplotlib import cm
from utils import vis,g_utils
import torch.nn.functional as F
Nums = 1
ifshow =1

def coords_grid(ht, wd):
    coords = torch.meshgrid(torch.arange(wd), torch.arange(ht))
    coords = torch.stack(coords[::-1], dim=2).float()
    ones = torch.ones((wd,ht,1)).float()
    cout = torch.cat([coords,ones],dim=2)
    return cout
def bilinear_sampler(img,coords , mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    img = img.permute(2,0,1).unsqueeze(0)
    coords =coords.unsqueeze(0)
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True,mode=mode)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img
def getRotation(x,y,z):
    x = torch.tensor(x)
    y = torch.tensor(y)
    z = torch.tensor(z)
    pi2j = torch.pi/180
    Rx = torch.eye(3)
    Rx[1,1] = torch.cos(x*pi2j)
    Rx[2,2] = torch.cos(x * pi2j)
    Rx[1,2] = -torch.sin(x * pi2j)
    Rx[2,1] = torch.sin(x * pi2j)

    Ry = torch.eye(3)
    Ry[0,0] = torch.cos(y*pi2j)
    Ry[2,2] = torch.cos(y * pi2j)
    Ry[0,2] = torch.sin(y * pi2j)
    Ry[2,0] = -torch.sin(y * pi2j)

    Rz = torch.eye(3)
    Rz[0,0] = torch.cos(z*pi2j)
    Rz[1,1] = torch.cos(z * pi2j)
    Rz[0,1] = -torch.sin(z * pi2j)
    Rz[1,0] = torch.sin(z * pi2j)

    Rxy = torch.matmul(Rx,Ry)
    Rxyz = torch.matmul(Rxy, Rz)
    return Rxyz
def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15 #值域为-512到512
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])
def writeRGBNERF(filename, rgb):
    rgb = (255.0 * rgb) #值域为0-32
    rgb = rgb.clip(0,255)
    rgbs = rgb.astype(np.uint8)
    rgbs = cv2.cvtColor(rgbs, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, rgbs)
def writeDDDCNERF(filename, d1,d2,dc):
    d1 = 1024.0 * d1 #值域为0-32
    d2 = 1024.0 * d2 #值域为0-32
    dc = 8192.0 * dc #值域为0-4
    dddc = np.concatenate([d1[:,:,np.newaxis],d2[:,:,np.newaxis], dc[:,:,np.newaxis]], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, dddc)

def writeMask(filename, ssim,d2dloss,alpha):
    ssim = 16384.0 * ssim #值域为0-2
    d2dloss = 16384.0 * d2dloss #值域为0-2
    alpha = 16384.0 * alpha #值域为0-2
    dddc = np.concatenate([ssim[:,:,np.newaxis],d2dloss[:,:,np.newaxis], alpha[:,:,np.newaxis]], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, dddc)
def get_ssim2(rgb_pred, rgb_gt,mask):
    """Evaluate the error between a predicted rgb image and the true image."""
    mask = mask[:,:,np.newaxis].astype(np.float32)
    rgb_predu = rgb_pred*mask
    rgb_gtu = rgb_gt*mask

    lploss =  np.abs(rgb_predu-rgb_gtu)
    rgb_predu = (rgb_predu * 255).astype(np.uint8)
    rgb_gtu = (rgb_gtu * 255).astype(np.uint8)
    rgb_pred_gray = cv2.cvtColor(rgb_predu, cv2.COLOR_RGB2GRAY)
    rgb_gt_gray = cv2.cvtColor(rgb_gtu, cv2.COLOR_RGB2GRAY)
    _,S = structural_similarity(rgb_pred_gray, rgb_gt_gray, data_range=255,full=True)
    sm = S[mask[:,:,0]>0].mean()

    rgb_pred = (rgb_pred * 255).astype(np.uint8)
    rgb_gt = (rgb_gt * 255).astype(np.uint8)
    rgb_pred_gray = cv2.cvtColor(rgb_pred, cv2.COLOR_RGB2GRAY)
    rgb_gt_gray = cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2GRAY)
    _, Su = structural_similarity(rgb_pred_gray, rgb_gt_gray, data_range=255, full=True)

    return sm,Su,lploss
def Trans1to2(p2cu,c2wn2u,c2wn1u,uv,depth1u):
    xy1 = torch.matmul(p2cu, uv.unsqueeze(3)).squeeze(3)
    xyzc = xy1 * depth1u  # TODO 像素投影到相机坐标系

    xyzc[:, :, 0] = -xyzc[:, :, 0]
    xyzc[:, :, 1] = -xyzc[:, :, 1]
    xyzc[:, :, 2] = -xyzc[:, :, 2]

    c2wn1u = torch.cat([c2wn1u, torch.zeros((1, 4))], dim=0)
    c2wn1u[3, 3] = 1  # 相机转世界坐标系

    xyzc1 = torch.cat([xyzc, torch.ones((uv.shape[0], uv.shape[1], 1))], dim=2)
    xyzw = torch.matmul(c2wn1u, xyzc1.unsqueeze(3))  # 转到世界坐标系

    c2wn2u = torch.cat([c2wn2u.cpu(), torch.zeros((1, 4))], dim=0)
    c2wn2u[3, 3] = 1
    w2c1n = torch.inverse(c2wn2u)  # 现在帧的像素转换

    xyzcn = torch.matmul(w2c1n, xyzw)  # 新视角下的相机坐标
    xyzcn[:, :, 0] = -xyzcn[:, :, 0]
    xyzcn[:, :, 1] = -xyzcn[:, :, 1]
    xyzcn[:, :, 2] = -xyzcn[:, :, 2]

    xy2n = xyzcn[:, :, :3, 0] / xyzcn[:, :, 2:3, 0]  # 除以深度后的相机坐标系坐标
    depthpn = xyzcn[:, :, 2:3, 0]
    c2p = torch.inverse(p2cu)
    uvn = torch.matmul(c2p, xy2n.unsqueeze(3)).squeeze(3)
    uvns = uvn.detach().cpu().numpy()  # 新视角下的坐标
    depthpns = depthpn[:, :, 0].detach().cpu().numpy()  # 新视角下的深度
    return xy2n,depthpn,uvn,uvns,depthpns
def getoccall(uv,flowf1,flowf1inv):
    uvn1 = uv[:, :, 0:2] + flowf1.permute(1, 2, 0).detach().cpu().numpy()
    flowu1 = flowf1inv.permute(1, 2, 0).detach().cpu()
    imout = bilinear_sampler(flowu1, uvn1[:, :, :2],mode='nearest')
    imoutu = imout[0] + flowf1.detach().cpu()

    occall = imoutu[0].detach().cpu().numpy()
    return occall
def viewtrans(view,c2w):
    view.R = c2w[0:3,0:3]
    view.T = c2w[0:3, 3]
    view.world_view_transform = torch.tensor(getWorld2View2(view.R, view.T, view.trans, 1)).transpose(0, 1).cuda()
    view.projection_matrix = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx,
                                                 fovY=view.FoVy).transpose(0, 1).cuda()
    view.full_proj_transform = (
        view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
    view.camera_center = view.world_view_transform.inverse()[3, :3]


    return view
def getnewpose(c2w):
    c2wn2 = torch.zeros_like(c2w.detach().cpu())
    c2wn3 = torch.zeros_like(c2w.detach().cpu())
    randxz = np.random.randint(-3, 3, 2)
    randy = np.random.randint(-3, 3, 1)
    #randxz[0] = 3
    #randxz[1] = 3

    directy = -c2w[:, 2:3].detach().cpu()
    directz = -c2w[:, 1:2].detach().cpu()
    directx = c2w[:, :1].detach().cpu()
    Rxyz = getRotation(randxz[0], randy[0], randxz[1])
    #Rxyz = getRotation(0, 0, 0)

    if randxz[0] > 0:
        zgo = -torch.rand(1) * 0.2 -0.05
    else:
        zgo = torch.rand(1) * 0.2 +0.05
    if randxz[1] > 0:
        xgo = -torch.rand(1) * 0.3 -0.05
    else:
        xgo = torch.rand(1) * 0.3 +0.05
    ygo = torch.rand(1) * 0.2-0.1

    c2wn2[:, :3] = torch.matmul(Rxyz, c2w[:, :3].detach().cpu())
    c2wn2[:, 3:] = c2w[:, 3:].detach().cpu() +xgo* directx + ygo * directy +zgo * directz
    c2wn2 = c2wn2.numpy()

    invRxyz = torch.inverse(Rxyz)
    c2wn3[:, :3] = torch.matmul(invRxyz, c2w[:, :3].detach().cpu())
    c2wn3[:, 3:] = c2w[:, 3:].detach().cpu() -xgo * directx - ygo * directy - zgo * directz
    c2wn3 = c2wn3.numpy()

    return c2wn2,c2wn3

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center

    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P
def getvis(depths):
    distance_limits = np.percentile(depths.flatten(), [0.8, 100 - 0.8])
    # lo, hi = [config.render_dist_curve_fn(x) for x in distance_limits]
    depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)
    lo, hi = distance_limits
    depths_vis = vis.visualize_cmap(depths, np.ones_like(depths), cm.get_cmap('turbo'), lo, hi,
                                    curve_fn=depth_curve_fn)
    return depths_vis
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor):
    #准备开始渲染光流数据，
    #指标可以新增一个正反光流一致性，涉及深度的话再加上一个几何结构一致性
    dataroot = '/home/lh/all_datasets/MIPGS10K_flow_forpaper'
    splitname = model_path.split('/')
    dataname = splitname[-2]
    dataroot = os.path.join(dataroot, dataname)
    #TODO 保存正向/反向光流的结果
    output_filename_flow = os.path.join(dataroot, 'flow/')
    if os.path.exists(output_filename_flow) == False:
        os.makedirs(output_filename_flow)
    output_filename_flowinv = os.path.join(dataroot, 'flow_inv/')
    if os.path.exists(output_filename_flowinv) == False:
        os.makedirs(output_filename_flowinv)
    # TODO 保存深度结果  第一帧，第二帧，深度变化率
    output_filenameDDDC = os.path.join(dataroot, 'depth/')
    if os.path.exists(output_filenameDDDC) == False:
        os.makedirs(output_filenameDDDC)
    #TODO 保存第一帧和第二帧图片结果
    output_filenamepic1 = os.path.join(dataroot, 'image1/')
    if os.path.exists(output_filenamepic1) == False:
        os.makedirs(output_filenamepic1)
    output_filenamepic2 = os.path.join(dataroot, 'image2/')
    if os.path.exists(output_filenamepic2) == False:
        os.makedirs(output_filenamepic2)
    # TODO 保存pah正与pah负，正反光流一致性（occall即遮挡掩膜）
    output_filenamePPF = os.path.join(dataroot, 'mask_PPF/')
    if os.path.exists(output_filenamePPF) == False:
        os.makedirs(output_filenamePPF)
    # TODO 保存掩膜 深度一致性掩膜，像素一致性掩膜，SSIM一致性掩膜
    output_filenameDPS = os.path.join(dataroot, 'mask_DPS/')
    if os.path.exists(output_filenameDPS) == False:
        os.makedirs(output_filenameDPS)
    output_filenamedepthvis = os.path.join(dataroot, 'depthvis/')
    if os.path.exists(output_filenamedepthvis) == False:
        os.makedirs(output_filenamedepthvis)
    kit = 0
    c2wlast = None
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):


        for viz in range(Nums):
            c2w = torch.from_numpy(np.concatenate([view.R, (view.T)[:, np.newaxis]], axis=-1)).float()
            ref_pose1 = view.world_view_transform.transpose(0, 1).inverse()
            kit = kit + 1
            renders = render(view, gaussians, pipeline, background, kernel_size=kernel_size)
            RGB = renders["render"].permute(1, 2, 0).detach().cpu().numpy()
            depth1 = renders["depthmid"].detach().cpu().numpy()
            #depth1m = renders["depthmean"].detach().cpu().numpy()
            pah = renders["pah"].detach().cpu().numpy()
            depths_vis = getvis(depth1)
            #depths_vism = getvis(depth1m)
            if ifshow:
                plt.imshow(RGB)
                plt.show()
                plt.imshow(depths_vis)
                plt.show()
                plt.imshow(pah)
                plt.show()
                maskocc = (pah<0.1).astype(np.uint8)*255
                maskoccs = np.concatenate([maskocc[:,:,np.newaxis],maskocc[:,:,np.newaxis],maskocc[:,:,np.newaxis]],axis=-1)
                plt.imshow(maskoccs)
                plt.show()
            depths_vis = 255.0 * depths_vis  # 值域为0-32
            depths_vis = depths_vis.astype(np.uint8)

            fileid = str(kit).zfill(5) + '.png'
            writeRGBNERF(os.path.join(output_filenamepic1, fileid), RGB)#保存初始帧的RGB
            cv2.imwrite(os.path.join(output_filenamedepthvis, fileid), depths_vis)#保存初始帧的可视化深度结果

            c2wf1, c2wf2 = getnewpose(c2w)  # 生成新的光流视角



            view = viewtrans(view, c2wf1)
            src_pose1 = view.world_view_transform.transpose(0, 1).inverse()
            renderF1 = render(view, gaussians, pipeline, background, kernel_size=kernel_size)
            RGBF1 = renderF1["render"].permute(1, 2, 0).detach().cpu().numpy()
            depth2 = renderF1["depthmid"].detach().cpu().numpy()
            pah2 = renderF1["pah"].detach().cpu().numpy()
            if ifshow:
                plt.imshow(RGBF1)
                plt.show()

            #TODO 下面开始准备各种掩膜
            uv = coords_grid( renderF1["render"].shape[2],  renderF1["render"].shape[1])

            mask1, depth_reprojected1, x2d_src1, y2d_src1, relative_depth_diff1, dist1,flowf1,sampled_rgb_src1,sampled_depth_src1,_,ddcout1 = g_utils.check_geometric_consistency(
                renders["depthmid"].detach().unsqueeze(0), view.K.cuda().detach().unsqueeze(0),
                ref_pose1.unsqueeze(0),RGB, renderF1["depthmid"].detach().unsqueeze(0),
                view.K.cuda().detach().unsqueeze(0), src_pose1.unsqueeze(0),RGBF1, thre1=2, thre2=0.01)
            mask1inv, depth_reprojected1inv, x2d_src1inv, y2d_src1inv, relative_depth_diff1inv, dist1inv,flowf1inv,sampled_rgb_src1inv,sampled_depth_src1inv,_,ddcout2 = g_utils.check_geometric_consistency(
                renderF1["depthmid"].detach().unsqueeze(0), view.K.cuda().detach().unsqueeze(0),
                src_pose1.unsqueeze(0),RGB, renders["depthmid"].detach().unsqueeze(0),
                view.K.cuda().detach().unsqueeze(0), ref_pose1.unsqueeze(0),RGBF1, thre1=2, thre2=0.01)

            dcchange1 = sampled_depth_src1[0].detach().cpu().numpy()/depth1
            #通过正反光流来计算遮挡部分
            occall = getoccall(uv,flowf1,flowf1inv)
            occall = abs(occall)/128
            rdds = relative_depth_diff1[0].detach().cpu().numpy()
            pds = dist1[0].detach().cpu().numpy()
            ddcs = ddcout1[0].detach().cpu().numpy()
            #用双向一致性来揭示遮挡,首先计算SSIM
            sampled_rgb_srcs = sampled_rgb_src1[0].permute(1,2,0).detach().cpu().numpy()
            ssim, S, lploss = get_ssim2(RGB, sampled_rgb_srcs, occall < 0.016)
            flows = flowf1.permute(1, 2, 0).detach().cpu().numpy()
            if ifshow:
                plt.imshow(sampled_rgb_srcs)
                plt.show()
                plt.imshow(ddcs<0.005)
                plt.show()
                plt.imshow(pah < 0.1)
                plt.show()
                plt.imshow(pah)
                plt.show()
                plt.imshow(vis.flow2rgb(flows))
                plt.show()
                plt.imshow(S>0.7)
                plt.show()

            #TODO 保存第二帧RGB  ，正反光流
            writeRGBNERF(os.path.join(output_filenamepic2, fileid), RGBF1)
            writeFlowKITTI(os.path.join(output_filename_flow, fileid), flowf1.permute(1,2,0).detach().cpu().numpy())
            writeFlowKITTI(os.path.join(output_filename_flowinv, fileid), flowf1.permute(1, 2, 0).detach().cpu().numpy())
            writeDDDCNERF(os.path.join(output_filenameDDDC, fileid), depth1, depth2, dcchange1)
            writeMask(os.path.join(output_filenamePPF, fileid), pah, pah2, occall)
            writeMask(os.path.join(output_filenameDPS, fileid), ddcs, rdds, S)
            print('FlowNum:', kit, 'SSIM_conf:', ssim, 'Disf', abs(flowf1.permute(1,2,0).detach().cpu().numpy()).mean())
            #因为3D高斯把场景描述为显式的三维高斯点云，它天生就有较好的多视角几何一致性，随机飞行噪声也更少
            #其次我们使用
            #下一步引入柔性3D高斯和误差容忍制度
            #双目网络可信度学习
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scale_factor = dataset.resolution
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size

        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, kernel_size, scale_factor=scale_factor)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    #args.resolution = 4
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)