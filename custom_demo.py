import os
import torch
import pdb

from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from multiprocessing import Process, Queue
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory
from dpvo.stream import image_stream, video_stream
from torch.utils.data import TensorDataset, DataLoader

# Hyperparameters
EPSILON = 255.0 # Max L-inf perturbation
LEARNING_RATE = 1e-3
PGD_EPOCHS = 20
BATCH_SIZE = 182  # Small batch for memory
NES_SAMPLES = 20  # Samples for gradient estimation
NES_NOISE_STD = 0.1  # Standard deviation for NES noise


@torch.no_grad()
def load_all_images(imagedir, calib, stride=1, skip=0):
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
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        frames.append(image)
        intrinsics_list.append(torch.from_numpy(intrinsics))

    reader.join()

    frames = torch.stack(frames)
    intrinsics_list = torch.stack(intrinsics_list)

    dataset = TensorDataset(frames, intrinsics_list)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    return dataloader

@torch.no_grad()
def run_slam(frames, intrinsics_list, slam):
    for idx in range(frames.shape[0]):
        t = idx
        image = frames[idx].cuda()
        intrinsics = intrinsics_list[idx].cuda()
        slam(t, image, intrinsics)

    (poses_array, tstamps_array) = slam.terminate()
    return poses_array, tstamps_array

@torch.no_grad()
def eval_slam(frames, intrinsics_list):
    _, H, W = frames_all.shape[1:]
    slam = DPVO(cfg, args.network, ht=H, wd=W, viz=args.viz)
    slam.network = slam.network.to(device)
    for p in slam.network.parameters():
        p.requires_grad_(False)

    for idx in range(frames.shape[0]):
        t = idx
        image = frames[idx].cuda()
        intrinsics = intrinsics_list[idx].cuda()
        slam(t, image, intrinsics)

    (poses_array, tstamps_array) = slam.terminate()
    return poses_array, tstamps_array


def compute_trajectory_loss(predicted, gt, align=True, correct_scale=False, weight_translation=1.0, weight_rotation=0.0):
    # Ensure tensors are on CPU and converted to NumPy
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    pred_traj = PoseTrajectory3D(
        positions_xyz=predicted[:, :3],
        orientations_quat_wxyz=predicted[:, [3, 4, 5, 6]],
        timestamps=np.arange(predicted.shape[0], dtype=np.float64)
    )
    gt_traj = PoseTrajectory3D(
        positions_xyz=gt[:, :3],
        orientations_quat_wxyz=gt[:, [3, 4, 5, 6]],
        timestamps=np.arange(gt.shape[0], dtype=np.float64)
    )

    gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    if align:
        pred_traj.align(gt_traj, correct_scale=correct_scale)

    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((gt_traj, pred_traj))
    translation_rmse = ape_metric.get_all_statistics()['rmse']

    rpe_metric = metrics.RPE(metrics.PoseRelation.rotation_angle_deg)
    rpe_metric.process_data((gt_traj, pred_traj))
    rotation_rmse = rpe_metric.get_all_statistics()['rmse']

    total_loss = (weight_translation * translation_rmse) + (weight_rotation * rotation_rmse)

    return torch.tensor(total_loss, device='cuda', requires_grad=False)


def nes_gradient(frames_batch, intrinsics_batch, noise_batch, gt_batch, slam, device):
    """Estimate gradient by Natural Evolution Strategies (NES)."""
    grads = torch.zeros_like(noise_batch)
    frames_batch = frames_batch.to(device)
    intrinsics_batch = intrinsics_batch.to(device)
    noise_batch = noise_batch.to(device)
    gt_batch = gt_batch.to(device)

    perturb = torch.randn((NES_SAMPLES, *noise_batch.shape), device=device)

    losses = torch.zeros(NES_SAMPLES, device=device)

    for j in range(NES_SAMPLES):
        adv_frames = torch.clamp(frames_batch + noise_batch + NES_NOISE_STD * perturb[j], 0, 255)
        pred_poses_pos, _ = run_slam(adv_frames, intrinsics_batch, slam)
        losses[j] = compute_trajectory_loss(pred_poses_pos, gt_batch).item()

    losses_standard = (losses - losses.mean()) / (losses.std() + 1e-8)

    perturb_flat = perturb.view(NES_SAMPLES, -1)  # Shape: (samples, flattened noise)
    grads_flat = torch.matmul(losses_standard, perturb_flat)  # Shape: (flattened noise,)
    grads = grads_flat.view_as(noise_batch)

    return grads


def pgd_attack(frames_loader, clean_poses, device):
    frames_all, intrinsics_all = frames_loader.dataset.tensors
    frames_all = frames_all.to(device)
    intrinsics_all = intrinsics_all.to(device)

    noise = torch.randn_like(frames_all, requires_grad=False).to(device)

    for step in range(PGD_EPOCHS):

        _, H, W = frames_all.shape[1:]
        slam = DPVO(cfg, args.network, ht=H, wd=W, viz=args.viz)
        slam.network = slam.network.to(device)
        for p in slam.network.parameters():
            p.requires_grad_(False)

        for batch_idx, (frames_batch, intrinsics_batch) in enumerate(frames_loader):
            start = batch_idx * frames_loader.batch_size
            end = start + frames_batch.size(0)

            frames_batch = frames_all[start:end]

            intrinsics_batch = intrinsics_all[start:end]
            noise_batch = noise[start:end]

            gt_batch = torch.from_numpy(clean_poses[start:end]).to(device).float()

            grad_estimate = nes_gradient(frames_batch, intrinsics_batch, noise_batch, gt_batch, slam, device)

            print(f"grad norm = {grad_estimate.norm()}")

            noise[start:end] += LEARNING_RATE/(NES_SAMPLES * NES_NOISE_STD) * grad_estimate

            noise[start:end] = torch.clamp(noise[start:end], -EPSILON, EPSILON)
        print(f"PGD Step {step} complete.")

    final_adv_frames = torch.clamp(frames_all + noise, 0, 255)
    return final_adv_frames


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--imagedir', type=str)
    parser.add_argument('--calib', type=str)
    parser.add_argument('--gt_trajectory', type=str)
    parser.add_argument('--name', type=str, default='result')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--plot', action="store_true")
    args = parser.parse_args()

    print("Running with config...")
    print(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print("Loading frames...")
    frame_loader = load_all_images(args.imagedir, args.calib, args.stride, args.skip)
    frames_all, intrinsics_all = frame_loader.dataset.tensors

    print("Loading ground truth trajectory...")
    gt_traj = file_interface.read_tum_trajectory_file(args.gt_trajectory)
    gt_poses = np.hstack((
        gt_traj.positions_xyz,  # x, y, z
        gt_traj.orientations_quat_wxyz[:, [1, 2, 3, 0]]  # qx, qy, qz, qw
    ))

    print("Running VO on clean frames...")
    clean_poses, tstamps = eval_slam(frames_all.cuda(), intrinsics_all.cuda())
    print(f"Clean loss = {compute_trajectory_loss(clean_poses, gt_poses)}")

    print("Running PGD Attack with NES approximation...")
    adv_frames = pgd_attack(frame_loader, gt_poses, device)

    adv_dir = Path("adv_frames")
    adv_dir.mkdir(exist_ok=True)

    adv_frames_clamped = adv_frames.clone() / 255.0

    for i, frame in enumerate(adv_frames_clamped):
        save_image(frame, adv_dir / f"adv_{i:05d}.png")

    print("Running VO on adversarial frames...")
    adv_poses, tstamps = eval_slam(adv_frames, intrinsics_all.cuda())

    print(f"Adv loss = {compute_trajectory_loss(adv_poses, gt_poses)}")

    trajectory = PoseTrajectory3D(
        positions_xyz=adv_poses[:, :3],
        orientations_quat_wxyz=adv_poses[:, [6, 3, 4, 5]],
        timestamps=tstamps
    )

    Path("saved_trajectories").mkdir(exist_ok=True)
    file_interface.write_tum_trajectory_file(f"saved_trajectories/{args.name}_adv.txt", trajectory)

    if args.plot:
        Path("trajectory_plots").mkdir(exist_ok=True)
        plot_trajectory(trajectory, title=f"Adversarial Trajectory {args.name}",
                        filename=f"trajectory_plots/{args.name}_adv.pdf")
