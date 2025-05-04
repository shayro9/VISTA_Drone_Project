import os
import pdb

import torch
import torch.nn.functional as F
from pathlib import Path
from multiprocessing import Process, Queue
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from dpvo.config import cfg
from dpvo.dpvo import DPVO

from dpvo.plot_utils import plot_trajectory
from dpvo.stream import image_stream, video_stream
from torch.utils.data import TensorDataset, DataLoader

# Hyperparameters for PGD
EPSILON = 255.0 / 255.0  # Max L-infinity norm perturbation
LEARNING_RATE = 1e-2  # Step size
PGD_EPOCHS = 10  # Number of PGD steps
BATCH_SIZE = 16


@torch.no_grad()
def load_all_images(imagedir, calib, stride=1, skip=0):
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

    frames = torch.stack(frames)  # (N, C, H, W)
    intrinsics_list = torch.stack(intrinsics_list)

    dataset = TensorDataset(frames, intrinsics_list)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    return dataloader


@torch.no_grad()
def eval_slam(frames, intrinsics_list, network, viz=False):
    """Run VO model on a list of frames (optionally perturbed)."""
    slam = None
    poses = []
    tstamps = []

    for idx in range(frames.shape[0]):
        t = idx
        image = frames[idx].cuda()
        intrinsics = intrinsics_list[idx].cuda()

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)

        slam(t, image, intrinsics)
        tstamps.append(t)

    # points = slam.pg.points_.cpu().numpy()[:slam.m]
    # colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]

    (poses_array, tstamps_array) = slam.terminate()

    return poses_array, tstamps_array


def run_slam(frames, intrinsics_list, network, viz=False):
    """Run VO model on a list of frames (optionally perturbed)."""
    slam = None
    poses = []
    tstamps = []

    for idx in range(frames.shape[0]):
        t = idx
        image = frames[idx].cuda()
        intrinsics = intrinsics_list[idx].cuda()

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)

        slam(t, image, intrinsics)
        tstamps.append(t)

    (poses_array, tstamps_array) = slam.terminate(return_tensor=True)

    return poses_array


def compute_trajectory_loss(predicted, gt, align=True, correct_scale=True, weight_translation=1.0, weight_rotation=1.0):
    """
    Compute trajectory loss based on APE (translation) and RPE (rotation).

    Args:
        predicted: np.ndarray of shape (N, 7) [x y z qw qx qy qz]
        gt: np.ndarray of shape (N, 7) [x y z qw qx qy qz]
    """
    # translation loss

    loss_trans = F.mse_loss(predicted[:, :3], gt[:, :3])
    # rotation loss (quaternion)
    pred_quat = predicted[:, 3:]
    gt_quat = gt[:, 3:]
    loss_rot = F.mse_loss(pred_quat, gt_quat)
    total_loss = loss_trans + 0.5 * loss_rot
    return -total_loss


def pgd_attack(frames_loader, network, gt_poses, device):
    """Perform PGD attack to maximize trajectory error."""
    frames_all, intrinsics_all = frames_loader.dataset.tensors
    frames_all = frames_all.to(device)
    intrinsics_all = intrinsics_all.to(device)

    # Initialize noise
    noise = torch.zeros_like(frames_all, requires_grad=True)

    optimizer = torch.optim.SGD([noise], lr=LEARNING_RATE)

    for step in range(PGD_EPOCHS):
        total_loss = 0.0

        for batch_idx, (frames_batch, intrinsics_batch) in enumerate(frames_loader):
            batch_size = frames_batch.size(0)
            start = batch_idx * frames_loader.batch_size
            end = start + batch_size

            frames_batch = frames_all[start:end]
            intrinsics_batch = intrinsics_all[start:end]
            noise_batch = noise[start:end]

            print("noise_batch grad_fn:", noise_batch.grad_fn)

            optimizer.zero_grad()

            # Add noise to batch and clamp to valid pixel range
            adv_frames = torch.clamp(frames_batch + noise_batch, 0, 255)

            print("adv_frames grad_fn:", adv_frames.grad_fn)

            # Run SLAM network
            pred_poses = run_slam(adv_frames, intrinsics_batch, network)

            # print("noise grad_fn:", pred_poses.grad_fn)

            # Prepare ground-truth poses for this batch
            pred_poses = pred_poses.data.to(device).float()

            print("pred_poses grad_fn:", pred_poses.grad_fn)

            gt_batch = torch.from_numpy(gt_poses[start:end]).to(device).float()


            # Compute trajectory loss
            loss = compute_trajectory_loss(pred_poses, gt_batch, weight_rotation=0.0)
            total_loss += loss.item()

            print("loss grad_fn:", loss.grad_fn)

            # Backpropagate and update noise
            loss.backward()
            optimizer.step()

            # Project noise into epsilon ball
            with torch.no_grad():
                noise[start:end] = torch.clamp(noise[start:end], -EPSILON * 255.0, EPSILON * 255.0)

        print(f"PGD Step {step}: Loss = {total_loss:.4f}")

    # Final adversarial frames
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

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    print("Running with config...")
    print(cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading frames...")
    frame_loader = load_all_images(args.imagedir, args.calib, args.stride, args.skip)
    frames_all, intrinsics_all = frame_loader.dataset.tensors

    print("Running VO on clean frames...")
    with torch.no_grad():
        clean_poses, tstamps = eval_slam(frames_all, intrinsics_all, args.network)

    # print("Loading ground truth trajectory...")
    # gt_traj = file_interface.read_tum_trajectory_file(args.gt_trajectory)
    # gt_poses = np.hstack((
    #     gt_traj.positions_xyz,  # x, y, z
    #     gt_traj.orientations_quat_wxyz[:, [1, 2, 3, 0]]  # qx, qy, qz, qw
    # ))

    print("Running PGD Attack...")
    adv_frames = pgd_attack(frame_loader, args.network, clean_poses, device=device)

    print("Running VO on adversarial frames...")
    adv_poses, _ = eval_slam(adv_frames, intrinsics_all, args.network)

    # Save and plot
    trajectory = PoseTrajectory3D(positions_xyz=adv_poses[:, :3], orientations_quat_wxyz=adv_poses[:, [6, 3, 4, 5]],
                                  timestamps=tstamps)
    Path("saved_trajectories").mkdir(exist_ok=True)
    file_interface.write_tum_trajectory_file(f"saved_trajectories/{args.name}_adv.txt", trajectory)

    if args.plot:
        Path("trajectory_plots").mkdir(exist_ok=True)
        plot_trajectory(trajectory, title=f"Adversarial Trajectory for {args.name}",
                        filename=f"trajectory_plots/{args.name}_adv.pdf")
