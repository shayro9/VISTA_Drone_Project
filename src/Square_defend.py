import bpy
import numpy as np
import subprocess
from Squere_class import SquareAttackLinfIterative
import os
from PIL import Image
import glob
import logging

SERVER_BASE = "shay.rozin@lambda.cs.technion.ac.il"
EPOCHS = 20

def apply_noise(image_dir, npy_file_path):
    # Load perturbations
    perturbations = np.load(npy_file_path).squeeze(0)  # Shape: (N, H, W, C)
    perturbations = np.transpose(perturbations, (0,2,3,1))
    
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
        image = bpy.data.images.new("TEXTURE_IMAGE", width=width, height=height, alpha=True, float_buffer=True, is_data=True)

    image.pixels = flat_pixels.tolist()
    image.update()
    
    image.filepath_raw = "./Desktop/results/test_output.png"
    image.file_format = 'PNG'
    image.save()
    print("Saved image to /tmp/test_output.png")
    
    image_node.image = image


# ============== Setup ===================
width, height = 256, 256  # Image size
base_image = np.zeros((1,3,height, width))
update_texture(base_image)
bpy.ops.render.render(animation=True)
square_class = SquareAttackLinfIterative(x=base_image, p_init=0.35, maximize=False)

logging.basicConfig(
    filename="./Desktop/results/loss_log.txt",     # Log file path
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w"                 # Overwrite each run; use "a" to append
)

# ============= Pipeline =================
# Render and send images
image_dir = bpy.context.scene.render.filepath
noise_path = "./Desktop/results/all_noise_S_Fly.npy"
remote_path = f"{SERVER_BASE}:~/drone/DPVO/input_output/images/clean_frames_S_Fly"
move_images_cmd = f"scp {image_dir}* {remote_path}"
subprocess.run(move_images_cmd, shell=True)

# Run pgd and calc noise
remote_path = f"{SERVER_BASE}:~/drone/DPVO/input_output/all_noise_S_Fly.npy"
local_path = f"./Desktop/results/all_noise_S_Fly.npy"

subprocess.run([
    "ssh", SERVER_BASE,
    "cd ~/drone/DPVO && srun -c 2 --gres=gpu:1 --pty ~/drone/DPVO/run_pgd.sh"
])

cmd = f"scp {remote_path} ./Desktop/results/"
subprocess.run(cmd, shell=True)

all_noise = np.load(local_path)

# Square defend iteration
for epoch in range(EPOCHS):
    perturbed_img = np.clip(square_class.get_pertubed_image(), 0, 1)
    update_texture(perturbed_img)

    bpy.ops.render.render(animation=True)
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    apply_noise(image_dir, noise_path)
    subprocess.run(move_images_cmd, shell=True)
    result = subprocess.run(["ssh", SERVER_BASE, "cd ~/drone/DPVO && srun -c 2 --gres=gpu:1 ~/drone/DPVO/run_no_attack.sh"],
                            text=True,
                            capture_output=True)
    loss = -float(result.stdout.strip())
    print(f"{epoch}: loss - ", loss)
    logging.info(f"Epoch {epoch}: Loss = {loss:.6f}")
    square_class.iterate(loss)
    subprocess.run(["ssh", SERVER_BASE, "cd ~/drone/DPVO && srun -c 2 --gres=gpu:1 ~/drone/DPVO/plot_traj.sh"])

