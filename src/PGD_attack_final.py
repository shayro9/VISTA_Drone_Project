import os
import pdb
import gc
import cv2

import torch
import torch.nn.functional as F
from pathlib import Path
from dpvo.lietorch import SE3
from torchvision.utils import save_image
from dpvo import projective_ops as pops
from multiprocessing import Process, Queue
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from dpvo.config import cfg
from dpvo.dpvo import DPVO
import numpy as np
from dpvo.plot_utils import plot_trajectory
from dpvo.stream import image_stream, video_stream
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast
from evo.core import metrics, sync

IN_OUT_DIR = 'input_output'

PGD_EPOCHS = 200
LEARNING_RATE = 0.25
BATCH_SIZE = 12
EPSILON = 15


def save_frames(frames, path):
    frames = frames.squeeze(0)  # Now shape is (n, c, h, w)
    n = frames.shape[0]

    # Directory to save images
    os.makedirs(path, exist_ok=True)

    # Save each frame
    for i in range(n):
        frame = frames[i]
        frame = frame.permute(1, 2, 0).numpy()
        cv2.imwrite(f'{path}/frame_{i:04d}.png', frame)

    print(f"Saved images to {path}")


def load_frames(imagedir, calib, stride=1, skip=0):
    """Load all frames into a tensor list."""
    frames = []
    intrinsics_list = []

    queue = Queue(maxsize=8)
    if os.path.isdir(imagedir):
        reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
    else:
        reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

    reader.start()

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0:
            break

        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (C, H, W)

        frames.append(image)
        intrinsics_list.append(torch.from_numpy(intrinsics))

    reader.join()

    frames = torch.stack(frames).unsqueeze(0)
    intrinsics_list = torch.stack(intrinsics_list).unsqueeze(0)

    return frames, intrinsics_list


def get_gt_poses(gt_dir):
    gt_traj = file_interface.read_tum_trajectory_file(f'{IN_OUT_DIR}/gt/{gt_dir}')

    gt_poses = torch.cat((
        torch.tensor(gt_traj.positions_xyz, dtype=torch.float32, device=device),
        torch.tensor(gt_traj.orientations_quat_wxyz[:, [1, 2, 3, 0]], dtype=torch.float32, device=device)
    ), dim=1)

    return gt_poses.unsqueeze(0)


def kabsch_umeyama(A, B):
    n, m = A.shape
    EA = torch.mean(A, axis=0)
    EB = torch.mean(B, axis=0)
    VarA = torch.mean((A - EA).norm(dim=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = torch.svd(H)

    c = VarA / torch.trace(torch.diag(D))
    return c


def calc_loss(traj):
    loss = 0.0
    for i, (v, x, y, P1, P2, kl) in enumerate(traj):
        e = (x - y).norm(dim=-1).unsqueeze(0)

        e = e.reshape(-1, 3 ** 2)[(v > 0.5).reshape(-1)].min(dim=-1).values

        N = P1.shape[1]
        ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
        ii = ii.reshape(-1).cuda()
        jj = jj.reshape(-1).cuda()

        k = ii != jj
        ii = ii[k]
        jj = jj[k]

        P1 = P1.inv()
        P2 = P2.inv()

        t1 = P1.matrix()[..., :3, 3]
        t2 = P2.matrix()[..., :3, 3]

        s = kabsch_umeyama(t2[0], t1[0]).clamp(max=10.0)
        P1 = P1.scale(s.view(1, 1))

        dP = P1[:, ii].inv() * P1[:, jj]
        dG = P2[:, ii].inv() * P2[:, jj]

        e1 = (dP * dG.inv()).log()
        tr = e1[..., 0:3].norm(dim=-1)
        ro = e1[..., 3:6].norm(dim=-1)

        loss += 0.1 * e.mean()
        if i >= 2:
            loss += 10 * (tr.mean() + ro.mean())

    return -loss


def create_model(network, H, W, device, viz):
    slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)
    slam.network = slam.network.to(device)
    slam.network = slam.network.float()
    for p in slam.network.parameters():
        p.requires_grad_(False)

    return slam


def calc_traj(frames, intrinsics, gt_poses, model, device):
    b, n, _, h, w = frames.shape
    disps = torch.ones(b, n, h, w, device="cuda")

    frames = frames.float()
    intrinsics = intrinsics.float()
    gt_poses = gt_poses.float()
    disps = disps.float()

    poses = SE3(gt_poses).inv()
    traj = model.network(frames, poses, disps, intrinsics, STEPS=BATCH_SIZE)

    return traj


def pgd_attack(frames_chunk, intrinsics_chunk, gt_poses_chunk, device, epsilon=EPSILON):
    frames_c = frames_chunk.to(device)
    intrinsics_c = intrinsics_chunk.to(device)
    gt_poses_c = gt_poses_chunk.to(device)

    # frames_c = frames_c / 255.0

    _, _, H, W = frames_c[0].shape
    model = create_model(args.network, H, W, device, args.viz)

    noise = torch.nn.Parameter(torch.zeros_like(frames_c) * 1.0, requires_grad=True)
    optimizer = torch.optim.Adam([noise], lr=LEARNING_RATE)

    for EPOCH in range(PGD_EPOCHS):
        optimizer.zero_grad()

        adv_frames = torch.clamp(frames_c + noise, 0.0, 255.0)

        traj = calc_traj(adv_frames, intrinsics_c, gt_poses_c, model, device)

        loss = calc_loss(traj)

        loss.backward()

        optimizer.step()

        with torch.no_grad():
            noise.data.clamp_(-epsilon, epsilon)
        if EPOCH % 50 == 0:
            print(f"Chunk attack epoch {EPOCH + 1}/{PGD_EPOCHS}, loss: {loss.item():.4f}")

    return noise.detach().cpu()


def run_no_attack(frames_chunk, intrinsics_chunk, gt_poses_chunk, device, epsilon=EPSILON):
    frames_c = frames_chunk.to(device)
    intrinsics_c = intrinsics_chunk.to(device)
    gt_poses_c = gt_poses_chunk.to(device)

    _, _, H, W = frames_c[0].shape
    model = create_model(args.network, H, W, device, args.viz)

    traj = calc_traj(frames_c, intrinsics_c, gt_poses_c, model, device)

    loss = calc_loss(traj)

    return loss


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--imagedir', type=str)
    parser.add_argument('--calib', type=str)
    parser.add_argument('--vid', action="store_true", default=True)
    parser.add_argument('--name', type=str, default='result')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--no_attack', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    name = args.name
    if not args.imagedir:
        imagedir = f'{IN_OUT_DIR}/videos/{name}.webm'
    else:
        imagedir = args.imagedir

    calib = f'{IN_OUT_DIR}/calib/S_Fly.txt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    frames, intrinsics = load_frames(imagedir, calib, args.stride, args.skip)
    ground_truth_poses = get_gt_poses(f'gt_{name}.txt')

    if args.no_attack:
        loss = run_no_attack(frames, intrinsics, ground_truth_poses, device)
        print(loss.item())
    else:
        if not args.imagedir:
            save_frames(frames, f'{IN_OUT_DIR}/images/clean_frames_{name}')

        all_noise = torch.zeros_like(frames)

        total_frames = frames[0].size(0)
        print(f"TOTAL_FRAMES =  {total_frames}")
        for start in range(0, total_frames, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_frames)
            if end < start + BATCH_SIZE: break
            print(f"Attacking frames {start} to {end - 1}")
            noise_chunk = pgd_attack(frames[:, start:end], intrinsics[:, start:end],
                                     ground_truth_poses[:, start:end], device)
            all_noise[:, start:end] = noise_chunk

        np.save(f'{IN_OUT_DIR}/all_noise_{name}.npy', all_noise.cpu().numpy())

        final_adv_frames = torch.clamp(frames + all_noise, 0.0, 255.0)

        save_frames(final_adv_frames, f'{IN_OUT_DIR}/images/noised_frames_{name}')
