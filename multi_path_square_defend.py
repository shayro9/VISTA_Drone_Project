import bpy
import numpy as np
import subprocess
from Squere_class import SquareAttackLinfIterative
import os
from PIL import Image
import glob
import logging
from get_camera_transform_blender_gt import calc_gt

SERVER_BASE = "shay.rozin@lambda.cs.technion.ac.il"
EPOCHS = 5
PATHS = 5

CAMERA_NAME = "Camera"


def apply_noise(image_dir, npy_file_path):
    # Load perturbations
    perturbations = np.load(npy_file_path).squeeze(0)  # Shape: (N, H, W, C)
    perturbations = np.transpose(perturbations, (0, 2, 3, 1))

    # Get sorted list of image paths
    image_dir_clean = image_dir.rstrip("/").removesuffix("frame_")
    image_paths = sorted(glob.glob(os.path.join(image_dir_clean, '*.png')))

    # Ensure matching count
    assert len(image_paths) == len(perturbations), "Number of images and perturbations must match"

    # Apply perturbations and overwrite original images
    for i, img_path in enumerate(image_paths):
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img).astype(np.float32)

        # Add perturbation
        perturbed = img_np + perturbations[i]
        perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)

        # Save back to original path (overwrite)
        Image.fromarray(perturbed).save(img_path)


def update_texture(np_img):
    light_obj = bpy.data.objects["Spot"]
    nodes = light_obj.data.node_tree.nodes
    image_node = None
    for node in nodes:
        if node.type == 'TEX_IMAGE':
            image_node = node
            break

    np_img = np_img[0].copy()

    # (3, W, H) → (W, H, 3)

    np_img = np.transpose(np_img, (1, 2, 0))

    # Ensure values are in [0, 1]
    if np_img.dtype != np.float32:
        np_img = np_img.astype(np.float32)
    if np_img.max() > 1.0:
        np_img /= 255.0

    height, width, _ = np_img.shape
    # Add alpha channel (1.0)
    alpha = np.ones((height, width, 1), dtype=np.float32)
    img_rgba = np.concatenate((np_img, alpha), axis=-1)

    # Flatten to 1D row-major for Blender
    flat_pixels = img_rgba.flatten()

    # Create or overwrite a Blender image
    if "TEXTURE_IMAGE" in bpy.data.images:
        image = bpy.data.images["TEXTURE_IMAGE"]
        image.scale(width, height)
    else:
        image = bpy.data.images.new("TEXTURE_IMAGE", width=width, height=height, alpha=True, float_buffer=True,
                                    is_data=True)

    image.pixels = flat_pixels.tolist()
    image.update()

    image.filepath_raw = "./Desktop/results/test_output.png"
    image.file_format = 'PNG'
    image.save()
    print("Saved image to /tmp/test_output.png")

    image_node.image = image


# ============== Setup ===================
width, height = 256, 256  # Image size
# base_image = np.ones((1, 3, height, width), dtype=np.uint8) * 255
base_image = np.zeros((1, 3, height, width))
update_texture(base_image)

cam = bpy.data.objects[CAMERA_NAME]
for i in range(PATHS):
    PATH_NAME = f"NurbsPath.00{i}"  # the curve object you want as Follow Path target

    path = bpy.data.objects[PATH_NAME]
    bpy.context.scene.render.filepath = f"/home/user2/Desktop/output_clean/path_{i}/frame_"

    fp = next((c for c in cam.constraints if c.type == 'FOLLOW_PATH'), None)
    if fp is None:
        fp = cam.constraints.new(type='FOLLOW_PATH')

    fp.target = path
    calc_gt(f"/home/user2/Desktop/results/output_clean/gt_path_{i}.txt")
    bpy.ops.render.render(animation=True)

square_class = SquareAttackLinfIterative(x=base_image, p_init=0.35, maximize=False)

logging.basicConfig(
    filename="./Desktop/results/loss_log.txt",  # Log file path
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w"  # Overwrite each run; use "a" to append
)

# ============= Pipeline =================
# Render and send images
for i in range(PATHS):
    image_dir = f"/home/user2/Desktop/output_clean/path_{i}/frame_"
    noise_path = f"./Desktop/results/all_noise_path_{i}.npy"
    remote_path = f"{SERVER_BASE}:~/drone/DPVO/input_output/images/clean_frames_path_{i}"
    move_images_cmd = f"scp {image_dir}* {remote_path}"
    subprocess.run(move_images_cmd, shell=True)

    remote_path = f"{SERVER_BASE}:~/drone/DPVO/input_output/gt/gt_path_{i}.txt"
    image_dir = f"/home/user2/Desktop/results/output_clean/gt_path_{i}.txt"
    move_gt_cmd = f"scp {image_dir}* {remote_path}"
    subprocess.run(move_gt_cmd, shell=True)

    # Run pgd and calc noise
    remote_path = f"{SERVER_BASE}:~/drone/DPVO/input_output/all_noise_path_{i}.npy"

    subprocess.run([
        "ssh", SERVER_BASE,
        f"cd ~/drone/DPVO && srun -c 2 --gres=gpu:1 --pty ~/drone/DPVO/run_pgd.sh path_{i}"
    ])

    cmd = f"scp {remote_path} ./Desktop/results/"
    subprocess.run(cmd, shell=True)

image_dir = f"/home/user2/Desktop/output_clean/path_0/frame_"
noise_path = f"./Desktop/results/all_noise_path_0.npy"
remote_path = f"{SERVER_BASE}:~/drone/DPVO/input_output/images/clean_frames_path_0"
move_images_cmd = f"scp {image_dir}* {remote_path}"

# Square defend iteration
min_loss = float('inf')
for epoch in range(EPOCHS):
    perturbed_img = np.clip(square_class.get_pertubed_image(), 0, 1)
    update_texture(perturbed_img)

    bpy.ops.render.render(animation=True)
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    apply_noise(image_dir, noise_path)
    subprocess.run(move_images_cmd, shell=True)
    result = subprocess.run(
        ["ssh", SERVER_BASE, "cd ~/drone/DPVO && srun -c 2 --gres=gpu:1 ~/drone/DPVO/run_no_attack.sh path_0"],
        text=True,
        capture_output=True)
    loss = -float(result.stdout.strip())
    min_loss = min(min_loss, loss)
    print(f"{epoch}: loss - ", loss)
    logging.info(f"Epoch {epoch}: Loss = {loss:.6f}")
    square_class.iterate(loss)
    if min_loss == loss:
        subprocess.run(["ssh", SERVER_BASE, "cd ~/drone/DPVO && srun -c 2 --gres=gpu:1 ~/drone/DPVO/plot_traj.sh path_0"])

for i in range(PATHS):
    PATH_NAME = f"NurbsPath.00{i}"  # the curve object you want as Follow Path target

    path = bpy.data.objects[PATH_NAME]
    bpy.context.scene.render.filepath = f"/home/user2/Desktop/output_clean/path_{i}/frame_"

    fp = next((c for c in cam.constraints if c.type == 'FOLLOW_PATH'), None)
    if fp is None:
        fp = cam.constraints.new(type='FOLLOW_PATH')

    fp.target = path

    bpy.ops.render.render(animation=True)

    image_dir = f"/home/user2/Desktop/output_clean/path_{i}/frame_"
    remote_path = f"{SERVER_BASE}:~/drone/DPVO/input_output/images/clean_frames_path_{i}"
    noise_path = f"./Desktop/results/all_noise_path_{i}.npy"
    move_images_cmd = f"scp {image_dir}* {remote_path}"
    apply_noise(image_dir, noise_path)
    subprocess.run(move_images_cmd, shell=True)
    subprocess.run(
        ["ssh", SERVER_BASE, f"cd ~/drone/DPVO && srun -c 2 --gres=gpu:1 ~/drone/DPVO/plot_traj.sh path_{i}"])
